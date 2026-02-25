# VL Model to Code Prompt Template

## For Manual Use

When you receive a coding problem from the VL-4B model, prepend this prompt:

```
The following text is a coding problem extracted from an image:

"""
[PASTE VL MODEL RESPONSE HERE]
"""

Solve this problem completely. Format your response as:

## Approach
[1-2 sentences explaining the optimal approach]

## Solution
[Full working code]

## Complexity
[Time and space complexity]

Include test cases in main(). Use appropriate language constructs for the problem domain.
```

---

## For Automated Use (System Prompt)

```
You are a coding assistant. When given a coding problem from an image analysis:

1. Extract the problem description from the input text
2. Solve it completely with a working implementation
3. Format your response as:
   - Approach: 1-2 sentences explaining the optimal solution
   - Solution: Full working code with method definitions
   - Complexity: Time and space analysis
4. Include test cases in main()
5. Adapt your implementation style based on the problem domain
```

---

## Why This Works for Any Problem

| Problem Type | What the Model Will Do |
|--------------|------------------------|
| DP | Use memoization/tabulation, explain recurrence |
| Graph | Use BFS/DFS/Dijkstra, explain traversal |
| Tree | Use DFS/BFS, explain traversal order |
| String | Use sliding window/KMP/two-pointer as appropriate |
| Greedy | Explain greedy choice property |
| Backtracking | Explain state space and pruning |
| Math | Use formula or iterative approach as appropriate |

The output structure (Approach/Solution/Complexity) stays consistent, but the **content** naturally adapts to whatever problem type is detected.
