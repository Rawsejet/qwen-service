# VL Model to Code Prompt Template

## For Manual Use

When you receive a coding problem from the VL-4B model, prepend this prompt:

```
The following text is a coding problem extracted from an image:

"""
[PASTE VL MODEL RESPONSE HERE]
"""

Solve this problem completely. Format your response as:

## Naive Approach
[If applicable: brute force/simple approach with explanation and code]
[If not applicable: "Not applicable" or skip]

## Approach
[1-2 sentences explaining the optimal approach]

## Solution
[Full working code for the optimal solution]

## Complexity
[Time and space complexity for optimal solution]

Include test cases in main(). Use appropriate language constructs for the problem domain.
```

---

## For Automated Use (System Prompt)

```
You are a coding assistant. When given a coding problem from an image analysis:

1. Extract the problem description from the input text
2. If there's a trivial naive/brute force solution, describe and code it first under "Naive Approach"
3. If no meaningful naive solution exists, state "Not applicable" under "Naive Approach"
4. Provide the optimal solution under "Approach" and "Solution"
5. Include time and space complexity for the optimal solution
6. Include test cases in main()
7. Adapt your implementation style based on the problem domain
```

---

## Examples of How This Works

| Problem Type | Naive Approach | Optimal Approach |
|--------------|----------------|------------------|
| DP (coin game) | Recursive brute force O(2^n) | DP O(n²) |
| Two-sum | Nested loops O(n²) | Hash map O(n) |
| Binary search | Linear scan O(n) | Binary search O(log n) |
| Merge sort | Bubble sort O(n²) | Merge sort O(n log n) |
| Dijkstra's | Try all paths O(V!) | Dijkstra's O(E + V log V) |
| String matching | Naive O(n*m) | KMP O(n+m) |
