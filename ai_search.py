"""
ai_search.py
TY B.Tech AIML Practical: Performance Evaluation of Search Algorithms
(Uninformed, Informed, Local Search, Constraint Satisfaction)

Run: python ai_search.py
"""

import time
import heapq
import math
from collections import deque

# ---------------- Campus Navigation Graph ----------------
# A=Gate B=Library C=Lab D=Canteen E=Admin F=Hostel G=Sports H=Auditorium
UNWEIGHTED = {
    "A": ["B", "D"], "B": ["A", "C", "D"], "C": ["B", "E", "G"],
    "D": ["A", "B", "E"], "E": ["C", "D", "F", "H"], "F": ["E", "H"],
    "G": ["C", "H"], "H": ["E", "F", "G"],
}
WEIGHTED = {
    "A": {"B": 4, "D": 2}, "B": {"A": 4, "C": 5, "D": 1},
    "C": {"B": 5, "E": 3, "G": 6}, "D": {"A": 2, "B": 1, "E": 7},
    "E": {"C": 3, "D": 7, "F": 2, "H": 5}, "F": {"E": 2, "H": 1},
    "G": {"C": 6, "H": 2}, "H": {"E": 5, "F": 1, "G": 2},
}
COORDS = {"A": (0, 0), "B": (2, 1), "C": (4, 1), "D": (2, 3),
          "E": (5, 4), "F": (7, 3), "G": (6, 0), "H": (8, 1)}

START, GOAL = "A", "H"


def h(n, goal):
    (x1, y1), (x2, y2) = COORDS[n], COORDS[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def cost_of(path):
    if not path or len(path) < 2:
        return 0
    return sum(WEIGHTED[u][v] for u, v in zip(path, path[1:]))


def result(name, found, path, nodes, t):
    return {"name": name, "found": found, "path": path or [], "cost": cost_of(path) if found else "N/A",
            "len": (len(path) - 1) if found else "N/A", "nodes": nodes, "time": t}


# ---------------- Uninformed Search ----------------
def bfs(s, g):
    t0 = time.perf_counter(); frontier = deque([[s]]); visited = {s}; n = 0
    while frontier:
        path = frontier.popleft(); node = path[-1]; n += 1
        for nb in UNWEIGHTED[node]:
            if nb not in visited:
                if nb == g:
                    return result("BFS", True, path + [nb], n, time.perf_counter() - t0)
                visited.add(nb); frontier.append(path + [nb])
    return result("BFS", False, None, n, time.perf_counter() - t0)


def dfs(s, g):
    t0 = time.perf_counter(); stack = [[s]]; visited = set(); n = 0
    while stack:
        path = stack.pop(); node = path[-1]
        if node in visited:
            continue
        visited.add(node); n += 1
        if node == g:
            return result("DFS", True, path, n, time.perf_counter() - t0)
        for nb in reversed(UNWEIGHTED[node]):
            if nb not in visited:
                stack.append(path + [nb])
    return result("DFS", False, None, n, time.perf_counter() - t0)


def ucs(s, g):
    t0 = time.perf_counter(); c = 0; frontier = [(0, c, [s])]; best = {s: 0}; n = 0
    while frontier:
        cost, _, path = heapq.heappop(frontier); node = path[-1]; n += 1
        if node == g:
            return result("UCS", True, path, n, time.perf_counter() - t0)
        for nb, w in WEIGHTED[node].items():
            nc = cost + w
            if nb not in best or nc < best[nb]:
                best[nb] = nc; c += 1
                heapq.heappush(frontier, (nc, c, path + [nb]))
    return result("UCS", False, None, n, time.perf_counter() - t0)


def dls(s, g, limit):
    t0 = time.perf_counter(); n = [0]

    def rec(path, depth):
        n[0] += 1
        if path[-1] == g:
            return path
        if depth >= limit:
            return None
        for nb in UNWEIGHTED[path[-1]]:
            if nb not in path:
                r = rec(path + [nb], depth + 1)
                if r:
                    return r
        return None

    r = rec([s], 0)
    return result(f"DLS(limit={limit})", r is not None, r, n[0], time.perf_counter() - t0)


def ids(s, g, max_depth):
    t0 = time.perf_counter(); total = 0
    for limit in range(max_depth + 1):
        n = [0]

        def rec(path, depth):
            n[0] += 1
            if path[-1] == g:
                return path
            if depth >= limit:
                return None
            for nb in UNWEIGHTED[path[-1]]:
                if nb not in path:
                    r = rec(path + [nb], depth + 1)
                    if r:
                        return r
            return None

        r = rec([s], 0); total += n[0]
        if r:
            return result("IDS", True, r, total, time.perf_counter() - t0)
    return result("IDS", False, None, total, time.perf_counter() - t0)


# ---------------- Informed Search ----------------
def greedy(s, g):
    t0 = time.perf_counter(); c = 0
    frontier = [(h(s, g), c, [s])]; visited = {s}; n = 0
    while frontier:
        _, _, path = heapq.heappop(frontier); node = path[-1]; n += 1
        if node == g:
            return result("Greedy", True, path, n, time.perf_counter() - t0)
        for nb in WEIGHTED[node]:
            if nb not in visited:
                visited.add(nb); c += 1
                heapq.heappush(frontier, (h(nb, g), c, path + [nb]))
    return result("Greedy", False, None, n, time.perf_counter() - t0)


def a_star(s, g):
    t0 = time.perf_counter(); c = 0
    frontier = [(h(s, g), c, 0, [s])]; best = {s: 0}; n = 0
    while frontier:
        f, _, gcost, path = heapq.heappop(frontier); node = path[-1]; n += 1
        if node == g:
            return result("A*", True, path, n, time.perf_counter() - t0)
        for nb, w in WEIGHTED[node].items():
            ng = gcost + w
            if nb not in best or ng < best[nb]:
                best[nb] = ng; c += 1
                heapq.heappush(frontier, (ng + h(nb, g), c, ng, path + [nb]))
    return result("A*", False, None, n, time.perf_counter() - t0)


# ---------------- Local Search: Hill Climbing ----------------
def objective(x):
    return -((x - 3) ** 2) + 10


def hill_climbing(start_x):
    t0 = time.perf_counter(); x = start_x; it = 0
    while True:
        neighbors = [x - 1, x + 1]
        best = max(neighbors, key=objective)
        it += 1
        if objective(best) > objective(x):
            x = best
        else:
            break
    return {"initial": start_x, "final": x, "value": objective(x), "iters": it,
            "time": time.perf_counter() - t0}


# ---------------- CSP: N-Queens (Backtracking) ----------------
def solve_n_queens(n):
    t0 = time.perf_counter()
    queens = [-1] * n
    stats = {"assign": 0, "backtrack": 0}

    def safe(row, col):
        for r in range(row):
            if queens[r] == col or abs(queens[r] - col) == abs(r - row):
                return False
        return True

    def backtrack(row):
        if row == n:
            return True
        for col in range(n):
            stats["assign"] += 1
            if safe(row, col):
                queens[row] = col
                if backtrack(row + 1):
                    return True
                queens[row] = -1
                stats["backtrack"] += 1
        return False

    found = backtrack(0)
    return {"n": n, "found": found, "solution": queens if found else None,
            "assign": stats["assign"], "backtrack": stats["backtrack"],
            "time": time.perf_counter() - t0}


# ---------------- Performance Comparison ----------------
def print_table(results):
    print(f"\n{'Algorithm':<16}{'Found':<8}{'Time(s)':<12}{'Nodes':<8}{'PathLen':<9}{'Cost':<6}")
    print("-" * 59)
    for r in results:
        print(f"{r['name']:<16}{str(r['found']):<8}{r['time']:<12.6f}{r['nodes']:<8}{str(r['len']):<9}{str(r['cost']):<6}")


def run_uninformed():
    return [bfs(START, GOAL), dfs(START, GOAL), ucs(START, GOAL),
            dls(START, GOAL, 3), ids(START, GOAL, 6)]


def run_informed():
    return [greedy(START, GOAL), a_star(START, GOAL)]


def menu():
    while True:
        print("\n===== AI SEARCH ALGORITHMS =====")
        print("1. Uninformed Search")
        print("2. Informed Search")
        print("3. Local Search")
        print("4. CSP - N Queens")
        print("5. Performance Comparison")
        print("6. Exit")
        ch = input("Enter choice (1-6): ").strip()

        if ch == "1":
            print_table(run_uninformed())
        elif ch == "2":
            print_table(run_informed())
        elif ch == "3":
            for sx in (-10, 0, 10):
                r = hill_climbing(sx)
                print(f"Start x={sx} -> Final x={r['final']}, f(x)={r['value']}, "
                      f"iters={r['iters']}, time={r['time']:.6f}s")
        elif ch == "4":
            for n in (4, 8):
                r = solve_n_queens(n)
                print(f"N={n}: found={r['found']}, assignments={r['assign']}, "
                      f"backtracks={r['backtrack']}, time={r['time']:.6f}s")
        elif ch == "5":
            results = run_uninformed() + run_informed()
            print_table(results)
        elif ch == "6":
            print("Exiting..."); break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    menu()
