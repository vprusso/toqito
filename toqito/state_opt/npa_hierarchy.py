"""Generates the NPA constraints."""

from collections import namedtuple
from itertools import product
import cvxpy

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])
IDENTITY_SYMBOL = Symbol("", None, None)
PLAYERS = ("Alice", "Bob")

def _reduce(word: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    if not word: return ()
    current_list = [s for s in word if s != IDENTITY_SYMBOL]
    if not current_list: return (IDENTITY_SYMBOL,) if any(s == IDENTITY_SYMBOL for s in word) else ()

    alice_ops = sorted([s for s in current_list if s.player == "Alice"], key=lambda s: (s.question, s.answer))
    bob_ops = sorted([s for s in current_list if s.player == "Bob"], key=lambda s: (s.question, s.answer))
    current_list = alice_ops + bob_ops

    while True:
        len_before_pass = len(current_list)
        next_pass_list, idx, made_change_in_pass = [], 0, False
        while idx < len(current_list):
            s_x = current_list[idx]
            if idx + 1 < len(current_list) and s_x.player == current_list[idx + 1].player and s_x.player in PLAYERS:
                s_y = current_list[idx + 1]
                if s_x == s_y:
                    next_pass_list.append(s_x)
                    idx += 2
                    made_change_in_pass = True
                    continue
                elif s_x.question == s_y.question:
                    return ()
            next_pass_list.append(s_x)
            idx += 1
        current_list = next_pass_list
        if not made_change_in_pass: break
    return tuple(current_list) if current_list else ()

def _parse(k_str: str) -> tuple[int, set[tuple[int, int]]]:
    if not k_str:
        raise ValueError("Input string k_str cannot be empty.")

    parts = k_str.split("+")

    # Validate and parse the base_k part
    if not parts[0]:
        raise ValueError("Base level k must be specified, e.g., '1+ab'")
    try:
        base_k = int(parts[0])
    except ValueError as e:
        raise ValueError(
            f"Base level k '{parts[0]}' is not a valid integer: {e}"
        ) from e

    # Validate and parse configuration parts
    configs = set()
    for i, part in enumerate(parts[1:]):
        if not part:  # Handle empty parts like "1++ab" or "1+"
            continue
        # Validate characters in the configuration string
        if not all(c in ("a", "b") for c in part):
            invalid_char = next((c for c in part if c not in ("a", "b")), None)
            raise ValueError(
                f"Invalid character '{invalid_char}' in k string component '{part}'. "
                "Only 'a' or 'b' allowed after base k."
            )
        configs.add((part.count("a"), part.count("b")))

    return base_k, configs

def _gen_words(k: int | str, a_out: int, a_in: int, b_out: int, b_in: int) -> list[tuple[Symbol, ...]]:
    # Use an independent basis: omit the last outcome for each question
    alice_symbols = [Symbol("Alice", x, a) for x, a in product(range(a_in), range(a_out - 1))]
    bob_symbols = [Symbol("Bob", y, b) for y, b in product(range(b_in), range(b_out - 1))]
    all_player_symbols = alice_symbols + bob_symbols
    
    words = {(IDENTITY_SYMBOL,)}
    k_int, configs = (k, set()) if isinstance(k, int) else _parse(k)

    for length in range(1, k_int + 1):
        for word_tuple in product(all_player_symbols, repeat=length):
            words.add(_reduce(word_tuple))
    for a_len, b_len in configs:
        for a_word in product(alice_symbols, repeat=a_len):
            reduced_a = _reduce(a_word)
            if not reduced_a and a_len > 0: continue
            for b_word in product(bob_symbols, repeat=b_len):
                reduced_b = _reduce(b_word)
                if not reduced_b and b_len > 0: continue
                words.add(_reduce(reduced_a + reduced_b))

    words.discard(())
    sorted_words = sorted(list(words - {(IDENTITY_SYMBOL,)}), key=lambda w: (len(w), str(w)))
    return [(IDENTITY_SYMBOL,)] + sorted_words

def _get_params(assemblage, dR):
    a_in, b_in = max(assemblage.keys())
    return assemblage[0,0].shape[0]//dR, a_in+1, assemblage[0,0].shape[1]//dR, b_in+1

def npa_constraints(assemblage, k=1, referee_dim=1):
    a_out, a_in, b_out, b_in = _get_params(assemblage, referee_dim)
    words = _gen_words(k, a_out, a_in, b_out, b_in)
    dim, dR = len(words), referee_dim
    word_to_idx = {word: i for i, word in enumerate(words)}
    
    moment_matrix = cvxpy.Variable((dR * dim, dR * dim), hermitian=True)
    constraints = [moment_matrix >> 0]
    
    rho_R = moment_matrix[0:dR, 0:dR]
    constraints.append(cvxpy.trace(rho_R) == 1)

    # Link moment matrix to the assemblage for basis operators
    seen_products = {}
    for i, word_i in enumerate(words):
        for j, word_j in enumerate(words):
            if i > j: continue
            block = moment_matrix[i*dR:(i+1)*dR, j*dR:(j+1)*dR]
            prod = _reduce(tuple(reversed(word_i)) + word_j)
            
            if not prod: constraints.append(block == 0); continue
            if prod in seen_products:
                p_i, p_j = seen_products[prod]
                constraints.append(block == moment_matrix[p_i*dR:(p_i+1)*dR, p_j*dR:(p_j+1)*dR]); continue
            
            seen_products[prod] = (i, j)
            if prod == (IDENTITY_SYMBOL,):
                constraints.append(block == rho_R)
            elif len(prod) == 2 and prod[0].player=="Alice" and prod[1].player=="Bob":
                s_a, s_b = prod; x, a, y, b = s_a.question, s_a.answer, s_b.question, s_b.answer
                constraints.append(block == assemblage[x, y][a*dR:(a+1)*dR, b*dR:(b+1)*dR])
            elif len(prod) == 1 and prod[0].player in PLAYERS:
                s = prod[0]; x, a = s.question, s.answer
                if s.player == "Alice":
                    constraints.append(block == sum(assemblage[x,0][a*dR:(a+1)*dR, b_idx*dR:(b_idx+1)*dR] for b_idx in range(b_out)))
                else:
                    constraints.append(block == sum(assemblage[0,x][a_idx*dR:(a_idx+1)*dR, a*dR:(a+1)*dR] for a_idx in range(a_out)))
    # No-signaling conditions:
    # 1. Alice's marginals must be independent of Bob's input `y`.
    for x_q in range(a_in):
        for a_ans in range(a_out):
            # Calculate Alice's marginal for the first Bob's input (y=0) as reference
            alice_marginal_ref = sum(
                assemblage[x_q, 0][a_ans * dR : (a_ans + 1) * dR, b_ans * dR : (b_ans + 1) * dR]
                for b_ans in range(b_out)
            )
            # Compare this reference marginal with marginals for other Bob's inputs `y_q_prime`.
            for y_q_prime in range(1, b_in):
                alice_marginal_current = sum(
                    assemblage[x_q, y_q_prime][
                        a_ans * dR : (a_ans + 1) * dR, b_ans * dR : (b_ans + 1) * dR
                    ]
                    for b_ans in range(b_out)
                )
                constraints.append(alice_marginal_ref == alice_marginal_current)

    # 2. Bob's marginals must be independent of Alice's input `x`.
    for y_q in range(b_in):
        for b_ans in range(b_out):
            # Calculate Bob's marginal for the first Alice's input (x=0) as reference
            bob_marginal_ref = sum(
                assemblage[0, y_q][a_ans * dR : (a_ans + 1) * dR, b_ans * dR : (b_ans + 1) * dR]
                for a_ans in range(a_out)
            )
            # Compare this reference marginal with marginals for other Alice's inputs `x_q_prime`.
            for x_q_prime in range(1, a_in):
                bob_marginal_current = sum(
                    assemblage[x_q_prime, y_q][
                        a_ans * dR : (a_ans + 1) * dR, b_ans * dR : (b_ans + 1) * dR
                    ]
                    for a_ans in range(a_out)
                )
                constraints.append(bob_marginal_ref == bob_marginal_current)

    # Add constraints for the dependent outcomes
    for x, y in product(range(a_in), range(b_in)):
        # Dependent Alice outcome
        a_last = a_out - 1
        sum_K_a = sum(assemblage[x,y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] for a in range(a_out-1) for b in range(b_out))
        K_a_last = sum(assemblage[x,y][a_last*dR:(a_last+1)*dR, b*dR:(b+1)*dR] for b in range(b_out))
        constraints.append(K_a_last == rho_R - sum_K_a)
        
        # Dependent Bob outcome
        b_last = b_out - 1
        sum_K_b = sum(assemblage[x,y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] for b in range(b_out-1) for a in range(a_out))
        K_b_last = sum(assemblage[x,y][a*dR:(a+1)*dR, b_last*dR:(b_last+1)*dR] for a in range(a_out))
        constraints.append(K_b_last == rho_R - sum_K_b)

    # Final assemblage positivity and normalization constraints
    for x, y in product(range(a_in), range(b_in)):
        for a, b in product(range(a_out), range(b_out)):
            constraints.append(assemblage[x, y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] >> 0)
        constraints.append(sum(assemblage[x,y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] for a,b in product(range(a_out),range(b_out))) == rho_R)

    return constraints
