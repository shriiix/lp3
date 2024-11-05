class Solution:
    def totalNQueens(self, n: int) -> int:
        answer = 0
        cols = set()
        posdiag = set()
        negdiag = set()
        board = [[" . "] * n for i in range(n)]

        def backtrack(i):
            nonlocal answer
            if i == n:
                answer += 1
                for row in board:
                    print("".join(row))
                print()
                return

            for j in range(n):
                if j in cols or (i + j) in posdiag or (i - j) in negdiag:
                    continue


                cols.add(j)
                posdiag.add(i + j)
                negdiag.add(i - j)
                board[i][j] = "Q"

                backtrack(i + 1)

                cols.remove(j)
                posdiag.remove(i + j)
                negdiag.remove(i - j)
                board[i][j] = " . "

        backtrack(0)
        return answer

obj = Solution()
print("Total Solutions:", obj.totalNQueens(4))