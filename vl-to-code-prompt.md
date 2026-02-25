# VL Model to Code Prompt Template

## For Manual Use

When you receive a coding problem from the VL-4B model, prepend this prompt:

```
The following text is a coding problem extracted from an image by a vision model:

"""
[PASTE VL MODEL RESPONSE HERE]
"""

Solve this problem completely. Follow this format:

## Problem Type
[1-2 sentences identifying the problem type and core concept]

## How to Solve
[1-2 sentences describing the optimal approach]

## Naive Approach
[Full working code with explanation]

## Better Approaches
[2-3 progressively better approaches with time/space complexity]

## Expected Input/Output
[Test cases based on the problem description]

Constraints:
- Use [language] language
- Include full class with main() and test case
- Use standard method names for the problem domain
- Optimize as appropriate for the problem type
```

---

## For Automated Use (System Prompt)

Add this as a system instruction when connecting VL-4B to a code model:

```
You are a coding assistant. When given a coding problem from an image analysis:

1. Extract the problem description from the input text
2. Identify the problem type (DP, Graph, Tree, String, Greedy, etc.)
3. Solve it completely with:
   - Problem type explanation (1-2 sentences)
   - Optimal solution approach (1-2 sentences)
   - Naive approach with full working code
   - 2-3 optimized approaches with complexity analysis
4. Use [language] with full class + main() + test cases
5. Use standard method names and appropriate variable conventions
6. Adapt your solution style based on the problem type:
   - DP: memoization or tabulation
   - Graph: BFS/DFS/Dijkstra/Floyd-Warshall
   - Tree: DFS/BFS with appropriate traversal
   - String: sliding window, two-pointer, KMP, etc.
   - Greedy/Backtracking: as appropriate
```

---

## Problem Type Guidelines (Internal Reference)

| Type | Approach | Common Patterns |
|------|----------|-----------------|
| DP | Memoization or Tabulation | i/j indices, dp table, recurrence relation |
| Graph | BFS/DFS/Dijkstra | adjacency list, visited set, queue/stack |
| Tree | DFS/BFS | preorder/inorder/postorder, recursion |
| String | Sliding window/Two-pointer | left/right pointers, hash map |
| Greedy | Local optimal | sort, iterate, accumulate |
| Backtracking | DFS with pruning | state, choice, constraint, recurse |
| Math | Formula/Pattern | loop, recurrence, direct formula |
| Binary Search | Search on answer | low/high, mid, condition |
