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
[1 sentence]

## How to Solve
[1-2 sentences]

## Naive Approach
[Full code]

## Better Approaches
[2-3 approaches with code, time/space complexity]

Constraints:
- Use [language] language
- Include full class with main() and test case
- Use standard method names for the problem
- For DP: use memoization or tabulation as appropriate
- Use common variable names (i, j, x, y, z, dp, etc.)
```

---

## For Automated Use (System Prompt)

Add this as a system instruction when connecting VL-4B to a code model:

```
You are a coding assistant. When given a coding problem from an image analysis:

1. Extract the problem description from the image text
2. Solve it completely with:
   - Problem type explanation (1 sentence)
   - Solution approach (1-2 sentences)
   - Naive approach with full code
   - 2-3 optimized approaches with complexity analysis
3. Use [language] with full class + main() + test case
4. Use standard method names and common variable conventions
```

---

## Quick Reference by Problem Type

### DP Problems
- Use `x=(i+2<=j)?dp[i+2][j]:0, y=(i+1<=j-1)?dp[i+1][j-1]:0, z=(i<=j-2)?dp[i][j-2]:0` pattern
- Tabulation with gap-based iteration

### Tree Problems
- Use DFS/BFS with standard traversal names

### Graph Problems
- Use BFS/DFS/Dijkstra based on problem requirements

### Array/String
- Two-pointer, sliding window, or hash map as appropriate
