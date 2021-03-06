## Leetcode
* 1-100:
  * [1. Two Sum](#1)
  * [2. Add Two Numbers](#2)
  * [3. Longest Substring Without Repeating Characters](#3)
  * [4. Median of Two Sorted Arrays](#4)
  * [5. Longest Palindromic Substring](#5)
  * [6. ZigZag Conversion](#6)
  * [7. Reverse Integer](#7)
  * [8. String to Integer (atoi)](#8)
  * [9. Palindrome Number](#9)
  * [10. Regular Expression Matching](#10)
  * [11. Container With Most Water](#11)
  * [12. Integer to Roman](#12)
  * [13. Roman to Integer](#13)
  * [14. Longest Common Prefix](#14)
  * [15. 3Sum](#15)
  * [16. 3Sum Closest](#16)
  * [17. Letter Combinations of a Phone Number](#17)
  * [18. 4Sum](#18)
  * [19. Remove Nth Node From End of List](#19)
  * [20. Valid Parentheses](#20)
  * [21. Merge Two Sorted Lists](#21)
  * [22. Generate Parentheses](#22)
  * [23. Merge k Sorted Lists](#23)
  * [24. Swap Nodes in Pairs](#24)
  * [25. Reverse Nodes in k-Group](#25)
  * [26. Remove Duplicates from Sorted Array](#26)
  * [27. Remove Element](#27)
  * [28. Implement strStr()](#28)
  * [30. Substring with Concatenation of All Words](#30)
  * [31. Next Permutation](#31)
  * [32. Longest Valid Parentheses](#32)
  * [33. Search in Rotated Sorted Array](#33)
  * [34. 在排序数组中查找元素的第一个和最后一个位置](#34)
  * [35. 搜索插入位置](#35)
  * [36. 有效的数独](#36)
  * [37. 解数独](#37)
  * [38. 外观数列](#38)
  * [39. 组合总和](#39)
  * [40. 组合总和 II](#40)
  * [75. 颜色分类](#75)
  * [94. 二叉树的中序遍历](#94)
  * [95. 不同的二叉搜索树 II](#95)
* 100+:
  * [101. 对称二叉树](#101)
  * [110. 平衡二叉树](#110)
  * [114. 二叉树展开为链表](#114)
  * [133. Clone Graph](#133)
  * [169. Majority Element](#169)
  * [207. Course Schedule](#207)
  * [210. Course Schedule II](#210)
  * [300. Longest Increasing Subsequence](#300)
  * [409. Longest Palindrome](#409)
  * [695. Max Area of Island](#695)
  * [794. Valid Tic-Tac-Toe State](#794)
  * [1071. Greatest Common Divisor of Strings](#1071)
  * [1160. Find Words That Can Be Formed by Characters](#1160)
* others
  * [interview1. The Minimum Number](#I1)
  * [interview2. 快手2020实习生招聘及补录-工程类笔试A卷-4](#I2)
  * [interview3. 阿里2020实习生招聘-1](#I3)
  * [interview4. 百度2020实习生招聘-1](#I4)
  * [interview5. 百度2020实习生招聘-2](#I5)
  * [interview6. 网易2020实习生招聘-1](#I6)
  * [interview7. 网易2020实习生招聘-2](#I7)

<a id="1"></a>
#### 1. Two Sum
 - Q: 找两个相加的数
 ````python
  Given nums = [2, 7, 11, 15], target = 9,
  Because nums[0] + nums[1] = 2 + 7 = 9,
  return [0, 1].
   ````
 - A:
  ````python
  class Solution:
      def twoSum(self, nums: List[int], target: int) -> List[int]:
          nums_cp = copy.deepcopy(nums)
          nums.sort()
          i = 0
          j = len(nums) - 1
          for k in range(len(nums)):
              if nums[i] + nums[j] == target:
                  index_i = nums_cp.index(nums[i])
                  index_j = nums_cp.index(nums[j])
                  if index_i == index_j:
                      index_j = len(nums_cp) - 1 - nums_cp[::-1].index(nums[j])
                  return [index_i, index_j]
              elif nums[i] + nums[j] > target:
                  j -= 1
              else:
                  i += 1
 ````

<a id="2"></a>
#### 2. Add Two Numbers
 - Q: 加两个链表数据
  ````python
  Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
  Output: 7 -> 0 -> 8
  Explanation: 342 + 465 = 807.

   ````
 - A:
  ````python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  class Solution:
      def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
          result = l1.val + l2.val
          l1.val = result % 10
          if not l1.next and not l2.next:
              if result >= 10:
                  l1.next = ListNode(1)
          elif not l1.next and l2.next:
              if result >= 10:
                  l1.next = self.addTwoNumbers(ListNode(1), l2.next)
              else:
                  l2.next.val += result // 10
                  l1.next = l2.next
          elif l1.next and not l2.next:
              if result >= 10:
                  l1.next = self.addTwoNumbers(l1.next, ListNode(1))
              else:
                  l1.next.val += result // 10
          else:
              l1.next.val += result // 10
              l1.next = self.addTwoNumbers(l1.next, l2.next)
          return l1
  ````

<a id="3"></a>
#### 3. Longest Substring Without Repeating Characters
 - Q: 寻找最长不重复子串
  ````python
  Input: "abcabcbb"
  Output: 3
  Explanation: The answer is "abc", with the length of 3.
  ````
 - A:
   - 解法一: 144 ms | 13.5 MB
  ````python
  class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == "":
            return 0
        begin = 0
        now = 1
        longest_substring = 1
        while(now < len(s)):
            for i in range(begin, now):
                if s[now] == s[i]:
                    begin = i + 1
                    break
                elif i == now - 1 and now - begin + 1 > longest_substring:
                    longest_substring = now - begin + 1
            now += 1

        return longest_substring
  ````
   - 解法二: 108 ms | 13.4 MB
   ````python
   class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hash_map = {}
        i, result = 0, 0
        for j in range(len(s)):
            if s[j] in hash_map:
                i = max(hash_map[s[j]], i)
            result = max(result, j - i + 1)
            hash_map[s[j]] = j + 1
        return result
   ````

<a id="4"></a>
#### 4. Median of Two Sorted Arrays
 - Q: 两个有序数组的中位数
 - A:
 ````python
 class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1 # n > m
        m = len(nums1)
        n = len(nums2)
        i_min = 0
        i_max = m
        while i_min <= i_max:
            i = (i_min + i_max) // 2
            j = (m + n + 1) // 2 - i
            if i > 0 and nums1[i - 1] > nums2[j]:
                i_max = i - 1
            elif i < m and nums2[j - 1] > nums1[i]:
                i_min = i + 1
            else:
            # i=0, j=0, i=m, j=n
                if (m + n) % 2 == 1:
                    if i == 0:
                        return nums2[j - 1]
                    elif j == 0:
                        return nums1[i - 1]
                    else:
                        return max(nums1[i - 1], nums2[j - 1])
                else:
                    k1 = 0 if i == 0 else nums1[i - 1]
                    k2 = 0 if j == 0 else nums2[j - 1]
                    k3 = nums2[j] if i == m else nums1[i]
                    k4 = nums1[i] if j == n else nums2[j]
                    return (max(k1, k2) + min(k3, k4)) / 2
 ````

<a id="5"></a>
#### 5. Longest Palindromic Substring
  - Q: 回文串
  ````python
  Input: "babad"
  Output: "bab"
  Note: "aba" is also a valid answer.
  ````
  - A:
  ````python
  class Solution:
    def longestPalindrome(self, s: str) -> str:
        if s == "":
            return ""
        for i in range(len(s), 1, -1):
            for j in range(0, len(s) - i + 1):
                mid = (2 * j + i - 1) // 2
                if i % 2 == 1:
                    if s[j: mid] == s[mid+1: j+i][::-1]:
                        return s[j: j+i]
                else:
                    if s[j: mid+1] == s[mid+1: j+i][::-1]:
                        return s[j: j+i]
        return s[0]
  ````

<a id="6"></a>
#### 6. ZigZag Conversion
 - Q: Z字型字符串
 ````python
  Input: s = "PAYPALISHIRING", numRows = 4
  Output: "PINALSIGYAHRPI"
  Explanation:

  P     I    N
  A   L S  I G
  Y A   H R
  P     I
 ````
 - A:
 ````python
  from collections import OrderedDict

  class Solution:
      def convert(self, s: str, numRows: int) -> str:
          if numRows == 1 or len(s) <= numRows:
              return s
          hash_map = OrderedDict({i:[] for i in range(numRows)})
          index = 0
          plus_num = 1
          for sub_char in s:
              hash_map[index].append(sub_char)
              if index == numRows - 1:
                  plus_num = -1
              elif index == 0:
                  plus_num = 1
              index += plus_num
          ans = ""
          for i in range(numRows):
              ans += "".join(hash_map[i])
          return ans
 ````

<a id="7"></a>
#### 7. Reverse Integer
 - Q: 数字翻转
 - A:
 ````python
 class Solution:
    def reverse(self, x: int) -> int:
        ans = abs(x)
        result = 0
        while ans != 0:
            nums = ans % 10
            result = result * 10 + nums
            ans = ans // 10
        if x < 0:
            if (result := -result) < -2 ** 31:
                return 0
        else:
            if result > 2 ** 31 - 1:
                return 0
        return result
 ````

<a id="8"></a>
#### 8. String to Integer (atoi)
  - Q: `atoi function`
  - A:
  ````python
  class Solution:
    def myAtoi(self, str: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', str.lstrip())), 2**31 - 1), -2**31)
  ````

<a id="9"></a>
#### 9. Palindrome Number
  - Q: 判断是不是回文串
  - A:
  ````python
  class Solution:
    def isPalindrome(self, x: int) -> bool:
        x = list(str(x))
        return x == x[::-1]
  ````

<a id="10"></a>
#### 10. Regular Expression Matching
  - Q: `.`和`*`的正则匹配
  ````python
  Input:
  s = "aa"
  p = "a"
  Output: false
  Explanation: "a" does not match the entire string "aa".

  Input:
  s = "ab"
  p = ".*"
  Output: true
  Explanation: ".*" means "zero or more (*) of any character (.)".
  ````
  - A:
  ````python
  class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        hash_map = {}
        def dp(i, j):
            if (i, j) not in hash_map:
                # 如果匹配, 结尾一定是j = len(p), i = len(s)
                if j == len(p):
                    ans = i == len(s)
                else:
                    # 当前字符是否匹配
                    single_match = i < len(s) and p[j] in (s[i], ".")
                    # *处只需要判断符合0次或者1次 + 前进一位继续判断
                    if j + 1 < len(p) and p[j + 1] == "*":
                        ans = single_match and dp(i + 1, j) or dp(i, j + 2)
                    else:
                        ans = single_match and dp(i + 1, j + 1)
                hash_map[i, j] = ans
            return hash_map[i, j]
        return dp(0, 0)
  ````

<a id="11"></a>
#### 11. Container With Most Water
  - Q: 求[A, B]之间的最大容量, 容量 = min(list[B], list[A]) * (B - A)
  ````python
  Input: [1,8,6,2,5,4,8,3,7]
  Output: 49
  ````
  - A:
  ````python
  class Solution:
    def maxArea(self, height: List[int]) -> int:
        length = len(height)
        front = 0
        end = length - 1
        ans = [0, length - 1, (length - 1) * min(height[0], height[length - 1])]
        while(front < end):
            if (end - front) * min(height[end], height[front]) > ans[2]:
                ans = [front, end, (end - front) * min(height[end], height[front])]
            if height[front] > height[end]:
                end -= 1
            else:
                front += 1
        return ans[2]
  ````

<a id="12"></a>
#### 12. Integer to Roman
  - Q: int转罗马数字
  - A:
     - 解法一:
     ````python
     class Solution:
     def intToRoman(self, num: int) -> str:
        ans = ""
        M_num = num // 1000
        ans += M_num * "M"
        num %= 1000
        D_num = num // 500
        num %= 500
        C_num = num // 100
        num %= 100
        if D_num == 1:
            ans += "CM" if C_num == 4 else "D" + C_num * "C"
        else:
            ans += "CD" if C_num == 4 else C_num * "C"
        L_num = num // 50
        num %= 50
        X_num = num // 10
        num %= 10
        if L_num == 1:
            ans += "XC" if X_num == 4 else "L" + X_num * "X"
        else:
            ans += "XL" if X_num == 4 else X_num * "X"
        V_num = num // 5
        num %= 5
        I_num = num
        if V_num == 1:
            ans += "IX" if I_num == 4 else "V" + I_num * "I"
        else:
            ans += "IV" if I_num == 4 else I_num * "I"
        return ans
     ````
     - 解法二: 贪心
     ````python
      class Solution:
      def intToRoman(self, num: int) -> str:
          ans = ""
          nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
          roman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
          index = 0
          while index < 13:
              while num - nums[index] >= 0:
                  ans += roman[index]
                  num -= nums[index]
              index += 1
          return ans
     ````

<a id="13"></a>
#### 13. Roman to Integer
  - Q: 罗马数字转int
  - A:
  ````python
  class Solution:
    def romanToInt(self, s: str) -> int:
        hash_map = {
            "M": 1000,
            "D": 500,
            "C": 100,
            "L": 50,
            "X": 10,
            "V": 5,
            "I": 1
        }
        ans = 0
        for i in range(len(s)):
            if i < len(s) - 1 and hash_map[s[i]] < hash_map[s[i + 1]]:
                ans -= hash_map[s[i]]
            else:
                ans += hash_map[s[i]]
        return ans
  ````

<a id="14"></a>
#### 14. Longest Common Prefix
  - Q:
  - A:
  ````python
  class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if strs == []:
            return ""
        ans = ""
        length = min([len(i) for i in strs])
        for i in range(length):
            s = strs[0][i]
            for j in range(len(strs)):
                if strs[j][i] != s:
                    return ans
            ans += s
        return ans
  ````

<a id="15"></a>
#### 15. 3Sum
  - Q: 三数之和
  ````python
  Given array nums = [-1, 0, 1, 2, -1, -4],

  A solution set is:
  [
    [-1, 0, 1],
    [-1, -1, 2]
  ]。
  ````
  - A: 就是从头开始固定第一个数, 寻找另外两个数 = -第一个数, 和2sum解法相同, 在遍历中有重复数字就直接跳过
  ````python
  class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        length = len(nums)
        if length < 3:
            return []
        nums.sort()
        for i in range(length):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            begin = i + 1
            end = length -1
            while begin < end:
                sum_num = nums[begin] + nums[end] + nums[i]
                if sum_num == 0:
                    ans.append([nums[i], nums[begin], nums[end]])
                    begin += 1
                    end -= 1
                    while begin < end and nums[begin] == nums[begin - 1]:
                        begin += 1
                    while end > begin and nums[end] == nums[end + 1]:
                        end -= 1
                elif sum_num > 0:
                    end -= 1
                else:
                    begin += 1
        return ans
  ````

<a id="16"></a>
#### 16. 3Sum Closest
  - Q: 找和最接近target的三个数
  ````python
  Given array nums = [-1, 2, 1, -4], and target = 1.
  The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
  ````
  - A:
  ````python
  class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        length = len(nums)
        if length < 3:
            return []
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(0, length - 2):
            if (temp:=nums[i] + nums[i + 1] + nums[i + 2]) > target:
                if abs(temp - target) > abs(ans - target):
                    return ans
                else:
                    return temp
            begin = i + 1
            end = length - 1
            while begin < end:
                sum_num = nums[i] + nums[begin] + nums[end]
                if sum_num == target:
                    return target
                if abs(sum_num - target) < abs(ans - target):
                    ans = sum_num
                if sum_num > target:
                    end -= 1
                else:
                    begin += 1
        return ans
  ````

<a id="17"></a>
#### 17. Letter Combinations of a Phone Number
  - Q: 九宫格拼音输入法
  ````python
  Input: "23"
  Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
  ````
  - A:
  ````python
  def letterCombinations(self, digits: str) -> List[str]:
    if digits == "":
        return []
    hash_map = {
        "2": [["a", "b", "c"], 3],
        "3": [["d", "e", "f"], 3],
        "4": [["g", "h", "i"], 3],
        "5": [["j", "k", "l"], 3],
        "6": [["m", "n", "o"], 3],
        "7": [["p", "q", "r", "s"], 4],
        "8": [["t", "u", "v"], 3],
        "9": [["w", "x", "y", "z"], 4],
    }
    ans = hash_map[digits[0]][0]
    for i in range(1, len(digits)):
        letters = hash_map[digits[i]][0]
        num = hash_map[digits[i]][1]
        ans *= num
        ans = [ans[j] + letters[j // (len(ans) // num)] for j in range(len(ans))]
    return ans
  ````

<a id="18"></a>
#### 18. 4Sum
  - Q:
  - A:
  ````python
  class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        length = len(nums)
        if length < 4:
            return []
        nums.sort()
        ans = []
        for i in range(0, length - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                return ans
            tri_target = target - nums[i]
            for j in range(i + 1, length - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                begin = j + 1
                end = length - 1
                while begin < end:
                    temp_sum = nums[j] + nums[begin] + nums[end]
                    if temp_sum == tri_target:
                        ans.append([nums[i], nums[j], nums[begin], nums[end]])
                        begin += 1
                        end -= 1
                        while begin < end and nums[begin] == nums[begin - 1]:
                            begin += 1
                        while begin < end and nums[end] == nums[end + 1]:
                            end -= 1
                    elif temp_sum > tri_target:
                        end -= 1
                    else:
                        begin += 1
        return ans
  ````

<a id="19"></a>
#### 19. Remove Nth Node From End of List
  - Q: 删除链表中的一个节点
  ````python
  Given linked list: 1->2->3->4->5, and n = 2.
  After removing the second node from the end, the linked list becomes 1->2->3->5.
  ````
  - A:
  ````python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None

  class Solution:
      def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
          temp_list = []
          num = 0
          l = ListNode(num)
          l.next = head
          while l.next:
              num += 1
              temp_list.append(l.next)
              l = l.next
          if num == n:
              return head.next
          temp_list[num - n - 1].next = temp_list[num - n].next
          return head
  ````

<a id="20"></a>
#### 20. Valid Parentheses
  - Q: 判断`()[]{}`组合的有效性, 栈的应用
  ````
  Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
  An input string is valid if:
  Open brackets must be closed by the same type of brackets.
  Open brackets must be closed in the correct order.
  Note that an empty string is also considered valid.
  ````
  - A:
  ````python
  class Solution:
    def isValid(self, s: str) -> bool:
        if s == "":
            return True
        stack = []
        hash_map = {
            "(": ")",
            "[": "]",
            "{": "}",
        }
        for i in range(len(s)):
            if len(stack) > (len(s) - i):
                return False
            if s[i] in hash_map:
                stack.append(hash_map[s[i]])
            elif stack == [] or s[i] != stack.pop():
                return False
        if stack != []:
            return False
        return True
  ````

<a id="21"></a>
#### 21. Merge Two Sorted Lists
  - Q: 按序合并两个链表
  ````python
  Input: 1->2->4, 1->3->4
  Output: 1->1->2->3->4->4
  ````
  - A:
  ````python
  class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val > l2.val:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
        else:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
  ````

<a id="22"></a>
#### 22. Generate Parentheses
  - Q: 找出括号数为n的最大有效组合
  ````python
  [
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
  ]
  ````
  - A:
  ````python
  class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return [""]
        ans = []
        def search(s, front, rear):
            if rear == 0 and front == 0:
                ans.append(s)
            if rear > front:
                search(s + ")", front, rear - 1)
            if front > 0:
                search(s + "(", front - 1, rear)
        search("(", n - 1, n)
        return ans
  ````

<a id="23"></a>
#### 23. Merge k Sorted Lists
  - Q: 合并k个有序链表, 双链表分治合并
  - A:
  ````python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  class Solution:
      def mergeKLists(self, lists: List[ListNode]) -> ListNode:
          ans = []
          for i in range(len(lists)):
              temp_list = lists[i]
              while temp_list:
                  ans.append(temp_list.val)
                  temp_list = temp_list.next
          if len(ans) == 0:
              return []
          ans.sort()
          head = ListNode(1)
          temp_list = ListNode(ans[0])
          head.next = temp_list
          for i in range(1, len(ans)):
              temp_list.next = ListNode(ans[i])
              temp_list = temp_list.next
          return head.next
  ````

<a id="24"></a>
#### 24. Swap Nodes in Pairs
  - Q: 链表中数据相互交换
  - A:
  ````python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  class Solution:
      def swapPairs(self, head: ListNode) -> ListNode:
          """
          :type head: ListNode
          :rtype: ListNode
          """
          # Dummy node acts as the prevNode for the head node
          # of the list and hence stores pointer to the head node.
          dummy = ListNode(-1)
          dummy.next = head

          prev_node = dummy

          while head and head.next:

              # Nodes to be swapped
              first_node = head;
              second_node = head.next;

              # Swapping
              prev_node.next = second_node
              first_node.next = second_node.next
              second_node.next = first_node

              # Reinitializing the head and prev_node for next swap
              prev_node = first_node
              head = first_node.next

          # Return the new head node.
          return dummy.next
  ````

<a id="25"></a>
#### 25. Reverse Nodes in k-Group
  - Q: 反转n个节点
  ````python
  Example:
  Given this linked list: 1->2->3->4->5
  For k = 2, you should return: 2->1->4->3->5
  For k = 3, you should return: 3->2->1->4->5
  ````
  - A:
  ````python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  class Solution:
      def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
          if not head.next:
              return head
          def k_reverse(head: ListNode):
              temp = ListNode(-1)
              temp.next = head
              for i in range(k):
                  if head:
                      head = head.next
                  else:
                      return temp.next
              head = temp.next
              end = head
              for i in range(k - 1):
                  temp = end.next
                  end.next = temp.next
                  temp.next = head
                  head = temp
              end.next = k_reverse(end.next)
              return head
          return k_reverse(head)
  ````

<a id="26"></a>
#### 26. Remove Duplicates from Sorted Array
  - Q: 去重
  - A:
  ````python
  class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 1
        length = len(nums)
        if length < 2:
            return length
        while i < length:
            if nums[i] == nums[i - 1]:
                del nums[i]
                length -= 1
            else:
                i += 1
        return len(nums)
  ````

<a id="27"></a>
#### 27. Remove Element
  - Q: 删除某个元素
  - A:
  ````python
  class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i, j = 0, 0
        length = len(nums)
        for i in range(length):
            if nums[i] == val:
                if j == 0:
                    j = i
                while j < length and nums[j] == val:
                    j += 1
                if j == length:
                    return i
                else:
                    nums[i], nums[j] = nums[j], nums[i]
        return length
  ````

<a id="28"></a>
#### 28. Implement strStr()
  - Q: 查找第一个匹配的子串
  ````python
  Input: haystack = "hello", needle = "ll"
  Output: 2
  ````
  - A:
  ````python
  class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        length = len(haystack)
        length2 = len(needle)
        if length2 == 0:
            return 0
        for i in range(length):
            if i + length2 > length:
                return -1
            elif haystack[i] == needle[0] and haystack[i: i + length2] == needle:
                return i
        return -1
  ````

<a id="30"></a>
#### 30. Substring with Concatenation of All Words
   - Q:
   ````
   给定一个字符串 s 和一些长度相同的单词 words。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
   注意子串要与 words 中的单词完全匹配，中间不能有其他字符，但不需要考虑 words 中单词串联的顺序。
   输入：
    s = "barfoothefoobarman",
    words = ["foo","bar"]
    输出：[0,9]
    解释：
    从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
    输出的顺序不重要, [9,0] 也是有效答案。
   ````
   - A:
   ````python
   一个数组匹配的hash, 遍历就行
   import collections
      class Solution:
          def findSubstring(self, s: str, words: List[str]) -> List[int]:
              res = []
              if not words:
                  return res
              word_len = len(words[0])
              win_len = word_len * len(words)
              word_cnt = collections.Counter(words)
              for left in range(word_len):
                  while left + win_len <= len(s):
                      temp = dict(word_cnt)
                      right = left + win_len
                      flag = True
                      while right > left:
                          cur_word = s[right - word_len: right]
                          if cur_word not in words or temp[cur_word] == 0:
                              flag = False
                              break
                          temp[cur_word] -= 1
                          right -= word_len
                      if not flag:
                          left = right
                          continue
                      res.append(left)
                      left += word_len
              return res
   ````

<a id="31"></a>
#### 31. Next Permutation
   - Q:
   ````
    实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
    如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
    必须原地修改，只允许使用额外常数空间。
    以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
    1,2,3 → 1,3,2
    3,2,1 → 1,2,3
    1,1,5 → 1,5,1
   ````
   - A:
   ````python
   设置两个标记位i, j, 从后向前遍历, 找到比i大且最接近i的数, 交换之后对i后面的进行升序排序
   class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        temp_nums = copy.deepcopy(nums)
        temp_nums.sort(reverse=True)
        if nums == temp_nums:
            nums.sort()
        else:
            flag = False
            for i in range(len(nums) - 2, -1, -1):
                ans = 10000
                index = len(nums)
                for j in range(len(nums) - 1, i, -1):
                    if nums[i] < nums[j] and nums[j] - nums[i] < ans:
                        ans = nums[i] - nums[j]
                        index = j
                if index != len(nums):
                    nums[index], nums[i] = nums[i], nums[index]
                    temp_nums = nums[i + 1:]
                    temp_nums.sort()
                    nums[i + 1:] = temp_nums
                    break
   ````

<a id="32"></a>
#### 32. Longest Valid Parentheses
   - Q:
   ````
   给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。
   示例:
    "()(()": 2
    "()()": 4
   ````
   - A:
   ````python
   利用栈, 初始化栈底元素为-1
   class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        ans = 0
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                j = stack.pop()
                if stack == []:
                    stack.append(i)
                else:
                    ans = ans if ans > i - stack[-1] else i - stack[-1]
        return ans
   ````

<a id="33"></a>
#### 33. Search in Rotated Sorted Array
   - Q:
   ````
     假设按照升序排序的数组在预先未知的某个点上进行了旋转。
    ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
    搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
    你可以假设数组中不存在重复的元素。
    你的算法时间复杂度必须是 O(log n) 级别。
    示例:
    输入: nums = [4,5,6,7,0,1,2], target = 0
    输出: 4
   ````
   - A:
   ````python
   先找到中间点, 再两边二分.
   class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def binary_search(arr, num):
            begin = 0
            end = len(arr) - 1
            while begin <= end:
                if begin == end:
                    return begin if arr[begin] == num else -1
                mid = (begin + end) // 2
                if arr[mid] == num:
                    return mid
                elif arr[mid] < num:
                    begin = mid + 1
                else:
                    end = mid
            return -1
        if nums == [] or target > nums[-1] and target < nums[0]:
            return -1
        if nums[0] < nums[-1]:
            return binary_search(nums, target)
        begin = 0
        end = len(nums) - 1
        j = 0
        while begin <= end:
            if begin == end:
                j = begin
                break
            mid = (begin + end) // 2
            if nums[mid] >= nums[0]:
                if nums[mid] > nums[mid + 1]:
                    j = mid + 1
                    break
                else:
                    begin = mid + 1
            else:
                if nums[mid] < nums[mid - 1]:
                    j = mid
                    break
                else:
                    end = mid
        if target <= nums[-1]:
            ans = binary_search(nums[j:], target)
            return ans if ans == -1 else ans + j
        else:
            return binary_search(nums[:j], target)
   ````

<a id="34"></a>
#### 34. 在排序数组中查找元素的第一个和最后一个位置
   - Q:
   ````python
    给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
    你的算法时间复杂度必须是 O(log n) 级别。
    如果数组中不存在目标值，返回 [-1, -1]。
    示例:
    输入: nums = [5,7,7,8,8,10], target = 8
    输出: [3,4]
   ````
   - A:
   ````python
   class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        first, second = -1, -1
        begin = 0
        end = len(nums) - 1
        flag = -1
        while(begin <= end):
            mid = (begin + end) // 2
            if nums[mid] == target:
                flag = mid
                break
            elif nums[mid] < target:
                begin = mid + 1
            else:
                end = mid - 1
        if flag == -1:
            return [-1, -1]
        if nums[0] == target:
            first = 0
        else:
            begin = 0
            end = flag
            while begin <= end:
                mid = (begin + end) // 2
                if nums[mid] == target:
                    if nums[mid - 1] < target:
                        first = mid
                        break
                    else:
                        end = mid - 1
                else:
                    begin = mid + 1
        if nums[-1] == target:
            second = len(nums) - 1
        else:
            begin = flag
            end = len(nums) - 1
            while begin <= end:
                mid = (begin + end) // 2
                if nums[mid] == target:
                    if nums[mid + 1] > target:
                        second = mid
                        break
                    else:
                        begin = mid + 1
                else:
                    end = mid - 1
        return [first, second]
   ````

<a id="35"></a>
#### 35. 搜索插入位置
   - Q:
   ````python
   给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，
   返回它将会被按顺序插入的位置。你可以假设数组中无重复元素。
   示例:
   输入: [1,3,5,6], 5
   输出: 2
   ````
   - A:
   ````python
   class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if nums == []:
            return 0
        begin = 0
        end = len(nums) - 1
        while begin <= end:
            mid = (begin + end) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                begin = mid + 1
            else:
                end = mid - 1
        if nums[mid] < target:
            return mid + 1
        else:
            return mid
   ````

<a id="36"></a>
#### 36. 有效的数独
   - Q:
   ````python
   判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
   数字 1-9 在每一行只能出现一次。
   数字 1-9 在每一列只能出现一次。
   数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
   输入:
    [
      ["5","3",".",".","7",".",".",".","."],
      ["6",".",".","1","9","5",".",".","."],
      [".","9","8",".",".",".",".","6","."],
      ["8",".",".",".","6",".",".",".","3"],
      ["4",".",".","8",".","3",".",".","1"],
      ["7",".",".",".","2",".",".",".","6"],
      [".","6",".",".",".",".","2","8","."],
      [".",".",".","4","1","9",".",".","5"],
      [".",".",".",".","8",".",".","7","9"]
    ]
   输出: true
   ````
   - A:
   ````python
   class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        n = len(board)
        column = [[] for i in range(n)]
        row = [[] for i in range(n)]
        block = [[] for i in range(n)]

        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == ".":
                    continue
                if board[i][j] in column[j] or board[i][j] in row[i] or board[i][j] in block[i // 3 * 3 + j // 3]:
                    return False
                else:
                    column[j].append(board[i][j])
                    row[i].append(board[i][j])
                    block[i // 3 * 3 + j // 3].append(board[i][j])
        return True
   ````

<a id="37"></a>
#### 37. 解数独
   - Q:
   ````python
   编写一个程序，通过已填充的空格来解决数独问题。一个数独的解法需遵循如下规则：
   数字 1-9 在每一行只能出现一次。
   数字 1-9 在每一列只能出现一次。
   数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
   空白格用 '.' 表示。
   ````
   - A:
   ````python
    from collections import defaultdict
    class Solution:
        def solveSudoku(self, board):
            """
            :type board: List[List[str]]
            :rtype: void Do not return anything, modify board in-place instead.
            """
            def could_place(d, row, col):
                """
                Check if one could place a number d in (row, col) cell
                """
                return not (d in rows[row] or d in columns[col] or \
                        d in boxes[box_index(row, col)])
            def place_number(d, row, col):
                """
                Place a number d in (row, col) cell
                """
                rows[row][d] += 1
                columns[col][d] += 1
                boxes[box_index(row, col)][d] += 1
                board[row][col] = str(d)
            def remove_number(d, row, col):
                """
                Remove a number which didn't lead
                to a solution
                """
                del rows[row][d]
                del columns[col][d]
                del boxes[box_index(row, col)][d]
                board[row][col] = '.'
            def place_next_numbers(row, col):
                """
                Call backtrack function in recursion
                to continue to place numbers
                till the moment we have a solution
                """
                # if we're in the last cell
                # that means we have the solution
                if col == N - 1 and row == N - 1:
                    nonlocal sudoku_solved
                    sudoku_solved = True
                #if not yet
                else:
                    # if we're in the end of the row
                    # go to the next row
                    if col == N - 1:
                        backtrack(row + 1, 0)
                    # go to the next column
                    else:
                        backtrack(row, col + 1)
            def backtrack(row = 0, col = 0):
                """
                Backtracking
                """
                # if the cell is empty
                if board[row][col] == '.':
                    # iterate over all numbers from 1 to 9
                    for d in range(1, 10):
                        if could_place(d, row, col):
                            place_number(d, row, col)
                            place_next_numbers(row, col)
                            # if sudoku is solved, there is no need to backtrack
                            # since the single unique solution is promised
                            if not sudoku_solved:
                                remove_number(d, row, col)
                else:
                    place_next_numbers(row, col)
            # box size
            n = 3
            # row size
            N = n * n
            # lambda function to compute box index
            box_index = lambda row, col: (row // n ) * n + col // n
            # init rows, columns and boxes
            rows = [defaultdict(int) for i in range(N)]
            columns = [defaultdict(int) for i in range(N)]
            boxes = [defaultdict(int) for i in range(N)]
            for i in range(N):
                for j in range(N):
                    if board[i][j] != '.':
                        d = int(board[i][j])
                        place_number(d, i, j)
            sudoku_solved = False
            backtrack()
   ````

<a id="38"></a>
#### 38. 外观数列
   - Q:
   ````python
   「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：
    1.     1
    2.     11
    3.     21
    4.     1211
    5.     111221
    1 被读作  "one 1"  ("一个一") , 即 11。
    11 被读作 "two 1s" ("两个一"）, 即 21。
    21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。
    给定一个正整数 n（1 ≤ n ≤ 30），输出外观数列的第 n 项。
    注意：整数序列中的每一项将表示为一个字符串。
    示例:
      输入: 4
      输出: "1211"
      解释：当 n = 3 时，序列是 "21"，其中我们有 "2" 和 "1" 两组，"2" 可以读作 "12"，
      也就是出现频次 = 1 而 值 = 2；类似 "1" 可以读作 "11"。所以答案是 "12" 和 "11"
      组合在一起，也就是 "1211"。
   ````
   - A:
   ````python
   class Solution:
    def countAndSay(self, n: int) -> str:
        string = "1"
        for i in range(n - 1):
            ans = ""
            char = string[0]
            num = 0
            for j in range(len(string)):
                if j == len(string) - 1:
                    if string[j] == char:
                        ans += str(num + 1) + string[j]
                    else:
                        ans += str(num) + string[j - 1] + "1" + string[j]
                    break
                if string[j] == char:
                    num += 1
                else:
                    char = string[j]
                    ans += str(num) + string[j - 1]
                    num = 1
            string = ans
        return string
   ````

<a id="39"></a>
#### 39. 组合总和
   - Q:
   ````python
    给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    candidates 中的数字可以无限制重复被选取。
    说明：
    所有数字（包括 target）都是正整数。
    解集不能包含重复的组合。 
    输入: candidates = [2,3,6,7], target = 7,
    所求解集为:
    [
      [7],
      [2,2,3]
    ]
   ````
   - A:
   ````python
   class Solution:
   def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
       if candidates == []:
           return []
       ans = []
       def search(already_arr, arr, target):
           if arr == []:
               return False
           num = target // arr[0]
           for i in range(num + 1):
               if arr[0] * i == target:
                   ans.append(already_arr + [arr[0]] * i)
               else:
                   search(already_arr + [arr[0]] * i, arr[1:], target - arr[0] * i)
       search([], candidates, target)
       return ans
   ````

<a id="40"></a>
#### 40. 组合总和 II
   - Q:
   ````python
    给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    candidates 中的每个数字在每个组合中只能使用一次。
    说明：
    所有数字（包括目标数）都是正整数。
    解集不能包含重复的组合。 
    示例 1:
    输入: candidates = [10,1,2,7,6,1,5], target = 8,
    所求解集为:
    [
      [1, 7],
      [1, 2, 5],
      [2, 6],
      [1, 1, 6]
    ]
   ````
   - A:
   ````python
   class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        if candidates == []:
            return []
        candidates.sort()
        ans = []
        size = len(candidates)
        def dfs(path, index, target):
            if target == 0:
                ans.append(path)
            for i in range(index, size):
                if candidates[i] > target:
                    return
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                dfs(path + [candidates[i]], i + 1, target - candidates[i])
        dfs([], 0, target)
        return ans
   ````

<a id="75"></a>
#### 75. 颜色分类
   - Q:
   ````python
    给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
    此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
    注意:
    不能使用代码库中的排序函数来解决这道题。
    示例:
    输入: [2,0,2,1,1,0]
    输出: [0,0,1,1,2,2]
    进阶：
    一个直观的解决方案是使用计数排序的两趟扫描算法。
    首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
    你能想出一个仅使用常数空间的一趟扫描算法吗？
   ````
   - A:
   ````python
   class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # nums.sort()
        begin = 0
        end = len(nums) - 1
        while begin < end:
            while begin < len(nums) - 1 and nums[begin] == 0:
                begin += 1
            while end > 0 and nums[end] != 0:
                end -= 1
            if begin < end:
                nums[begin], nums[end] = nums[end], nums[begin]
        end = len(nums) - 1
        while begin < end:
            while begin < len(nums) - 1 and nums[begin] == 1:
                begin += 1
            while end > 0 and nums[end] == 2:
                end -= 1
            if begin < end:
                nums[begin], nums[end] = nums[end], nums[begin]
   ````

<a id="94"></a>
#### 94. 二叉树的中序遍历
   - Q:
   ````python
    给定一个二叉树，返回它的中序 遍历。
    示例:
    输入: [1,null,2,3]
       1
        \
         2
        /
       3
    输出: [1,3,2]
    进阶: 递归算法很简单，你可以通过迭代算法完成吗？
   ````
   - A:
   ````python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    class Solution:
        def __init__(self):
            self.ans = []
        def inorderTraversal(self, root: TreeNode) -> List[int]:
            def search(tree):
                if not tree:
                    return
                if not tree.left and not tree.right:
                    self.ans.append(tree.val)
                    return
                if not tree.left:
                    self.ans.append(tree.val)
                    search(tree.right)
                    return
                if not tree.right:
                    search(tree.left)
                    self.ans.append(tree.val)
                    return
                search(tree.left)
                self.ans.append(tree.val)
                search(tree.right)
            search(root)
            return self.ans
   ````

<a id="95"></a>
#### 95. 不同的二叉搜索树 II
   - Q:
   ````python
    给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。
    示例:
    输入: 3
    输出:
    [
      [1,null,3,2],
      [3,2,null,1],
      [3,1,null,null,2],
      [2,1,3],
      [1,null,2,null,3]
    ]
    解释:
    以上的输出对应以下 5 种不同结构的二叉搜索树：
       1         3     3      2      1
        \       /     /      / \      \
         3     2     1      1   3      2
        /     /       \                 \
       2     1         2                 3
   ````
   - A:
   ````python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    class Solution:
        def generateTrees(self, n: int) -> List[TreeNode]:
            ans = []
            nodes = [i + 1 for i in range(n)]
            def build_bst(tree, left_arr, right_arr):
                temp = []
                if left_arr != []:
                    left_tree = []
                    for i in range(len(left_arr)):
                        left_tree += build_bst(left_arr[i], left_arr[:i], left_arr[i + 1:])
                else:
                    left_tree = [None]
                if right_arr != []:
                    right_tree = []
                    for i in range(len(right_arr)):
                        right_tree += build_bst(right_arr[i], right_arr[:i], right_arr[i + 1:])
                else:
                    right_tree = [None]

                for left_node in left_tree:
                    for right_node in right_tree:
                        temp_tree = TreeNode(tree)
                        temp_tree.left = left_node
                        temp_tree.right = right_node
                        temp.append(temp_tree)
                return temp
            for i in nodes:
                ans += build_bst(i, nodes[:i - 1], nodes[i:])
            return ans
   ````

<a id="101"></a>
#### 101. 对称二叉树
   - Q:
   ````python
    给定一个二叉树，检查它是否是镜像对称的。
    例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
        1
       / \
      2   2
     / \ / \
    3  4 4  3
    但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
        1
       / \
      2   2
       \   \
       3    3
    进阶：
    你可以运用递归和迭代两种方法解决这个问题吗？
   ````
   - A:
   ````python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    # class Solution:
    #     def isSymmetric(self, root: TreeNode) -> bool:
    #         if not root:
    #             return True
    #         def symmetric(tree1 , tree2):
    #             if tree1 == tree2:
    #                 return True
    #             if not tree1 or not tree2 or tree1.val != tree2.val:
    #                 return False
    #             return symmetric(tree1.right, tree2.left) and symmetric(tree1.left, tree2.right)
    #         return symmetric(root.left, root.right)
    from collections import deque
    class Solution:
        def isSymmetric(self, root: TreeNode) -> bool:
            if not root or root.left == root.right:
                return True
            queue1 = deque()
            queue1.append(root.left)
            queue2 = deque()
            queue2.append(root.right)
            while queue1 and queue2:
                tree1 = queue1.popleft()
                tree2 = queue2.popleft()
                if tree1 == tree2:
                    continue
                if not tree1 or not tree2 or tree1.val != tree2.val:
                    return False
                queue1.append(tree1.left)
                queue1.append(tree1.right)
                queue2.append(tree2.right)
                queue2.append(tree2.left)
            if queue1 == queue2:
                return True
            else:
                return False
   ````

<a id="110"></a>
#### 110. 平衡二叉树
   - Q:
   ````python
    给定一个二叉树，判断它是否是高度平衡的二叉树。
    本题中，一棵高度平衡二叉树定义为：
    一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
    示例 1:
    给定二叉树 [3,9,20,null,null,15,7]
        3
       / \
      9  20
        /  \
       15   7
    返回 true 。
    示例 2:
    给定二叉树 [1,2,2,3,3,null,null,4,4]
           1
          / \
         2   2
        / \
       3   3
      / \
     4   4
    返回 false 。
   ````
   - A:
   ````python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    class Solution:

        def __init__(self):
            self.flag = True

        def isBalanced(self, root: TreeNode) -> bool:
            if not root:
                return True
            def balance(tree):
                if not tree.left and not tree.right:
                    return 0
                if not tree.left:
                    height = balance(tree.right) + 1
                    if height > 1:
                        self.flag = False
                    return height
                if not tree.right:
                    height = balance(tree.left) + 1
                    if height > 1:
                        self.flag = False
                    return height
                height1 = balance(tree.left) + 1
                height2 = balance(tree.right) + 1
                if abs(height1 - height2) > 1:
                    self.flag = False
                return height1 if height1 > height2 else height2
            balance(root)
            return self.flag
   ````

<a id="114"></a>
#### 114. 二叉树展开为链表
   - Q:
   ````python
    给定一个二叉树，原地将它展开为链表。
    例如，给定二叉树
        1
       / \
      2   5
     / \   \
    3   4   6
    将其展开为：
    1
     \
      2
       \
        3
         \
          4
           \
            5
             \
   ````
   - A:
   ````python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    from collections import deque
    class Solution:
        def flatten(self, root: TreeNode) -> None:
            """
            Do not return anything, modify root in-place instead.
            """
            if not root:
                return None
            def merge(tree):
                if not tree.left and not tree.right:
                    return tree
                if not tree.left:
                    tree.right = merge(tree.right)
                    return tree
                if not tree.right:
                    tree.right = merge(tree.left)
                    tree.left = None
                    return tree
                left_node = merge(tree.left)
                right_node = merge(tree.right)
                tree.right = left_node
                tree.left = None
                while left_node.right != None:
                    left_node = left_node.right
                left_node.right = right_node
                return tree
            merge(root)
   ````

<a id="133"></a>
#### 133. Clone Graph
   - Q: 邻接表遍历图
   - A:
     - 解法一: DFS
     ````python
      """
      # Definition for a Node.
      class Node:
          def __init__(self, val = 0, neighbors = []):
              self.val = val
              self.neighbors = neighbors
      """
      class Solution:
          def cloneGraph(self, node: 'Node') -> 'Node':
              if not node:
                  return None
              gragh_map = {}
              def search_node(node):
                  if node.val not in gragh_map:
                      gragh_map[node.val] = [nei_node.val for nei_node in node.neighbors]
                      for i in node.neighbors:
                          search_node(i)
              search_node(node)
              nodes = [0]
              for i in range(1, len(gragh_map) + 1):
                  nodes.append(Node(i))
              for i in range(1, len(gragh_map) + 1):
                  nodes[i].neighbors = [nodes[j] for j in gragh_map[i]]
              return nodes[1]
     ````
     - 解法二: BFS
     ````python
      """
      # Definition for a Node.
      class Node:
          def __init__(self, val = 0, neighbors = []):
              self.val = val
              self.neighbors = neighbors
      """
      from collections import deque
      class Solution:
          def cloneGraph(self, node: 'Node') -> 'Node':
              if not node:
                  return None
              queue = deque([node])
              hash_map = {}
              hash_map[node] = Node(node.val)
              while queue:
                  temp_node = queue.popleft()
                  if temp_node.neighbors:
                      for neighbor in temp_node.neighbors:
                          if neighbor not in hash_map:
                              hash_map[neighbor] = Node(neighbor.val)
                              queue.append(neighbor)
                          hash_map[temp_node].neighbors.append(hash_map[neighbor])
              return hash_map[node]
     ````

<a id="169"></a>
#### 169. Majority Element
  - Q: 求众数
  ````python
  Input: [3,2,3]
  Output: 3
  ````
  - A:
     - 解法一:
    ````python
    from collections import Counter
    class Solution:
      def majorityElement(self, nums: List[int]) -> int:
          c = Counter(nums)
          return c.most_common(1)[0][0]
    ````
     - Boyer-Moore 投票
     ````python
     from collections import Counter
     class Solution:
        def majorityElement(self, nums: List[int]) -> int:
           ans = nums[0]
           num = 0
           for i in nums:
               if num == 0:
                   ans = i
                   num = 1
               else:
                   num += 1 if ans == i else -1
           return ans
     ````

<a id="207"></a>
#### 207. Course Schedule
   - Q:
   - A:
     - 解法一: DFS
     ````python
     class Solution:
      def __init__(self):
          self.temp_arr = []
      def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
          if not prerequisites:
              return True
          # 建立邻接表
          # 遍历所有节点, DFS, 记录每一个可行的节点
          # 记录path, 当形成回环时返回false
          adjacency_matrices = [[] for i in range(numCourses)]
          for i, j in prerequisites:
              adjacency_matrices[i].append(j)
          ans = []
          def search_end(node, path):
              if node not in ans and adjacency_matrices[node]:
                  if node in path:
                      return False
                  path.append(node)
                  for i in adjacency_matrices[node]:
                      if not search_end(i, path):
                          return False
              ans.append(node)
              return True
          for i in range(numCourses):
              if i not in ans and not search_end(i, []):
                  return False
          return True
     ````
     - 解法二: 拓扑排序
     ````python
     from collections import deque
        class Solution:
            def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
                if not prerequisites:
                    return True
                # 拓扑排序, 建立一个队列, 不断的删除入度为0的节点
                adajence_matrix = [[] for i in range(numCourses)]
                queue = deque()
                ans = [0] * numCourses
                for i, j in prerequisites:
                    adajence_matrix[j].append(i)
                    ans[i] += 1
                for i in range(numCourses):
                    if ans[i] == 0:
                        queue.append(i)
                while queue:
                    for node_number in adajence_matrix[queue.popleft()]:
                        ans[node_number] -= 1
                        if ans[node_number] == 0:
                            queue.append(node_number)
                if sum(ans) == 0:
                    return True
                else:
                    return False
     ````

<a id="210"></a>
#### 210. Course Schedule II
   - Q: 排课拓扑排序
   - A:
   ````python
    from collections import deque
    class Solution:
        def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
            if numCourses == 0:
                return []
            adjacency_matrix = [[] for i in range(numCourses)]
            ans = [0] * numCourses
            queue = deque()
            for i, j in prerequisites:
                adjacency_matrix[j].append(i)
                ans[i] += 1
            for i in range(numCourses):
                if ans[i] == 0:
                    queue.append(i)
            result = []
            while queue:
                pop_number = queue.popleft()
                result.append(pop_number)
                for node_number in adjacency_matrix[pop_number]:
                    ans[node_number] -= 1
                    if ans[node_number] == 0:
                        queue.append(node_number)
            if sum(ans) != 0:
                return []
            else:
                return result
   ````

<a id="300"></a>
#### 300. Longest Increasing Subsequence
  - Q:
  - A:
     - 解法一:DP
     ````python
     class Solution:

     def lengthOfLIS(self, nums: List[int]) -> int:
        if nums == []:
            return 0
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
        return max(dp)
     ````
     - 解法二: 贪心+二分查找
     ````python
     class Solution:
     # 存储每个点的最大序列, 后面点的最大序列根据这个点之前所有点的组合生成
     def lengthOfLIS(self, nums: List[int]) -> int:
        if nums == []:
            return 0
        ans = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] > ans[-1]:
                ans.append(nums[i])
            else:
                begin = 0
                end = len(ans) - 1
                mid = (begin + end) // 2
                while begin < end:
                    if nums[i] > ans[mid]:
                        begin = mid + 1
                    else:
                        end = mid
                    mid = (begin + end) // 2
                if nums[i] > ans[mid]:
                    ans[mid + 1] = nums[i]
                else:
                    ans[mid] = nums[i]
        return len(ans)
     ````

<a id="409"></a>
#### 409. Longest Palindrome
   - Q: 最长回文串
   - A:
   ````python
    from collections import Counter
    class Solution:
        def longestPalindrome(self, s: str) -> int:
            if s == "":
                return 0
            hash_map = Counter(s)
            ans = 0
            for key, value in hash_map.items():
                ans += hash_map[key] // 2
            if (ans := ans * 2) < len(s):
                ans += 1
            return ans
   ````

<a id="695"></a>
#### 695. Max Area of Island
  - Q: 寻找最大连接的岛屿数量
  ````
 [[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
 最大岛屿连接岛屿为6
  ````
  - A:
  ````python
  class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ans = []
        height = len(grid)
        width = len(grid[0])

        def search_grid(i, j):
            if i < 0 or j < 0 or i == height or j == width or grid[i][j] == 0:
                return 0
            else:
                grid[i][j] = 0
                return 1 + search_grid(i - 1, j)+  search_grid(i, j - 1) + search_grid(i + 1, j) + search_grid(i, j + 1)

        for i in range(height):
            for j in range(width):
                if grid[i][j] == 1:
                    ans.append(search_grid(i, j))
        if ans == []:
            return 0
        return max(ans)
  ````

<a id="794"></a>
#### 794. Valid Tic-Tac-Toe State
   - Q: 三子棋
   - A:
   ````python
   class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        len_x = 0
        len_o = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    len_x += 1
                elif board[i][j] == 'O':
                    len_o += 1
        if len_x < len_o or len_x > len_o + 1:
            return False
        if len_x + len_o > 9:
            return False
        flag_x = 0
        flag_y = 0
        mode = ["XXX"]
        for i in range(3):
            if board[i] in mode:
                flag_x += 1
        for j in range(3):
            if board[0][j] + board[1][j] + board[2][j] in mode:
                flag_x +=1
        if board[0][0] + board[1][1] + board[2][2] in mode:
            flag_x += 1
        if board[0][2] + board[1][1] + board[2][0] in mode:
            flag_x += 1

        mode = ["OOO"]
        for i in range(3):
            if board[i] in mode:
                flag_y += 1
        for j in range(3):
            if board[0][j] + board[1][j] + board[2][j] in mode:
                flag_y += 1
        if board[0][0] + board[1][1] + board[2][2] in mode:
            flag_y += 1
        if board[0][2] + board[1][1] + board[2][0] in mode:
            flag_y += 1

        if flag_x == 1 and flag_y == 1:
            return False
        elif flag_x == 1 and flag_y == 0:
            if len_x == len_o:
                return False
        elif flag_x == 0 and flag_y == 1:
            if len_x > len_o:
                return False
        return True
   ````

<a id="1071"></a>
#### 1071. Greatest Common Divisor of Strings
  - Q: 求最大公因子
  ````python
  Input: str1 = "ABCABC", str2 = "ABC"
  Output: "ABC"
  ````
  - A:
  ````python
  class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if len(str1) < len(str2):
            str1, str2 = str2, str1
        if str2 != str1[: len(str2)]:
            return ""
        k = 1
        n = len(str2)
        while(k < n):
            base = n // k
            if str2[: base] * k == str2 and str2[: base] * (len(str1) // base) == str1:
                return str2[:base]
            k += 1
        return ""
  ````

<a id="1160"></a>
#### 1160. Find Words That Can Be Formed by Characters
   - Q: 查询`words`中`["cat","bt","hat","tree"]`那些可以被`chars`:`"atach"`组合
   - A:
   ````python
   class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        chars_cnt = collections.Counter(chars)
        ans = 0
        for word in words:
            word_cnt = collections.Counter(word)
            for c in word_cnt:
                if chars_cnt[c] < word_cnt[c]:
                    break
            else:
                ans += len(word)
        return ans
   ````

<a id="I1"></a>
#### interview1. The Minimum Number
   - Q:
   ````python
   含有重复数的数组，求这个数组的子集合的和中，最小不能得到的数,
   比如[1,2,5]. 最小不能得到的是4
   ````
   - A:
   ````python
    arr = [1, 2, 3, 4, 5]
    max_number = sum(arr)
    result = [0] * (max_number + 1)
    result[0] = max_number + 1
    for i in arr:
        result[i] += 1

    def min_number(arr, count):
        if sum(result) == 2 * max_number + 1:
            return max_number + 1
        if not arr:
            return result.index(0)
        front_num = result.index(0)
        if min(arr) > front_num:
            return front_num
        result[arr[0] + count] = 1
        if len(arr) == 1:
            return min_number([], arr[0] + count)
        else:
            ans = [min_number(arr[1:], arr[0] + count)]
            for i in range(1, len(arr)):
                if arr[i] != arr[i - 1]:
                    if i == len(arr) - 1:
                        result[arr[-1] + count] = 1
                        ans.append(min_number(arr[:-1], arr[-1] + count))
                    else:
                        result[arr[i] + count]
                        ans.append(min_number(arr[:i] + arr[i + 1:], arr[i] + count))
        return max(ans)
    print(min_number(arr, 0))
   ````

<a id="I2"></a>
#### interview2. 快手2020实习生招聘及补录-工程类笔试A卷-4
   - Q:
   ````
   保护伞芯片公司现在拥有一块特殊的芯片材料板,这块材料板可以看做一个
   n*m大小的矩阵,矩阵中的每一个单元蕴含着不的能量。现在公司需要放置
   若干块ab大小的芯片在这块材料板上,但是放芯片的前提是材料板有a*b大
   小的空位置,并且这些位置的能量是一样的。为了满足矩形内每个单元的能
   量是一样的这个要求,公司需要释放某些单格一定量的能量释放的量可以精
   确控制。释放能量的代价是e1-e2(其中e>=e2e1表示释放前的能量e2表示
   释放之后的能量)。放置一块芯片的代为矩形内释放的总能量之和。(保护伞
   公司技术限制,能量只能释放出去而不能注入进去)现在公司按照如下策略去
   放置芯片:1.找到可以放置下1块芯片所需要释放的最小能量,然后放置该
   芯片(程序输出中用芯片的左上角坐标记录)2.如果有多个位置代价一样,那
   么按照行小的先,如果行一样,按照列小的优先. 3.重复1步骤继续下一个芯
   片直到整个面板无再放置芯片。注意,芯片之间不能重叠。现在你可以帮助公
   司计算出可以放置芯片的总数和他们的位置及代价吗?
   输入:
    第一行有4个正整数n,m,a,b分别表示n*m大小的材料板和a*b大小的芯片
    其中1<=a<=n<=10001<=b<=m<=1000下面n行每行有m个数字,表示对应
    位置的能量每个单位的能量值不超过1,000,000,000。测试数据中33的数
    据满足1<=n,m<=100。
   输出描述:
    输出包含k行,k表示按照公司的放置策略可以放置的芯片数量。下面k行每
    行包含3个数字R1,,Cost分别表示放置第块芯片左上角的行,列以及放置
    这个芯片的代价(需要释放的总能量之和,请注意数据范围不要溢出)
   示例:
    输入:
      2 2 2 1
      2 1
      3 1
    输出:
      2
      1 2 0
      1 1 1
   ````
   - A:
   ````python
    核心思想就是k表示每一个放上去的方块, k的总数是一定的, 按序放置的动态规划
    n, m, a, b = [int(i) for i in input().split()]
    kernel = []
    for i in range(n):
        kernel.append([int(i) for i in input().split()])
    row_bias = n - n // a * a
    column_num = m // b
    row_num = n // a
    row_arr = [row_bias] * column_num
    ans = []
    def search_min(k, column_bias, row_arr):
        if k > column_num * row_num - 1:
            return 0
        if k % column_num == 0:
            column_bias = m - m // b * b
        temp_row_bias = row_arr[k % column_num]
        y = k // column_num * a + (row_bias - temp_row_bias)
        x = k % column_num * b + (m - m // b * b - column_bias)
        min_x, min_y, min_cost = [-1] * 3
        for i in range(temp_row_bias + 1):
            for j in range(column_bias + 1):
                row_arr[k % column_num] -= i
                kernel_slice = [kernel[q][x: x + j + b] for q in range(y, y + i + a)]
                min_slice = min(min(kernel_slice))
                sum_slice = sum([sum(q) for q in kernel_slice])
                slice_cost = sum_slice - min_slice * a * b
                temp_cost = search_min(k + 1, column_bias - j, row_arr)
                if min_cost == -1 or slice_cost + temp_cost < min_cost:
                    min_cost = temp_cost + slice_cost
                    min_x, min_y = x, y
        ans.append([min_y + 1, min_x + 1, min_cost])
        return min_cost
    search_min(0, m - m // b * b, row_arr)
    print(len(ans))
    print(ans)
   ````

<a id="I3"></a>
#### interview3. 阿里2020实习生招聘-1
   - Q:
   ````
   现有n个人,要从这n个人中选任意数量的人组成一只队伍,再在这些人中选出一名队长,求
   不同的方案数对10+7取模的结果。如果两个方案选取的人的集合不同或选出的队长不同,
   则认为这两个方案是不同的。
   ````
   - A:
   ````python
   快速幂
    # n * 2^(n - 1)
      n = 5
      M = 10 ** 9 - 7
      result = 1
      ans = n - 1
      a = 2
      while ans:
          if ans & 1 == 1:
              result *= a % M
          a = a * a % M
          ans >>= 1
      print(n * result % M)
   ````

<a id="I4"></a>
#### interview4. 百度2020实习生招聘-1
   - Q:
   ````
   今天，度度熊和牛妹在玩取石子的游戏，开始的时候有几堆石头，第1堆有4个石头，两个人轮流动作，
   度度熊先走，在每个回合，玩家选择一个非空堆，并从堆中移除-块石头。 如果一个玩家在轮到他之
   前所有的石碓都是空的，或者如果在移动石头之后，存在两个堆包含相同数量的石头(可能为都为0)，
   那么他就会输。假设两人都在游戏时选择最佳方式，度度熊和牛妹谁会赢?如果度度熊获胜，输出"man",
   如果牛妹获胜，输出"woman" (输出不包含双引号)
   输入描述:
    第一行一个数表示T组数据(1≤T<10),
    每组数据第一行一个数n,表示n堆石头(1≤n≤10^5)
    第二行n个数，表示每堆中石头的个数。(0≤ai≤10^9)
   输出描述:
    每组一行如果度度熊获胜，输出“man", 如果牛妹获胜，输出"woman" (输出不包含双引号)
   示例1:
    输入:
      2
      1
      1
    输出:
      "man"
    ````
   - A:
    ````python
    核心思想: 轮到自己时就选自己能赢的情况
    import copy
      def judge(arr):
          hash_map = {}
          for i in arr:
              if i in hash_map:
                  return True
              hash_map[i] = 1
          return False
      def who_win(arr, sex):
          if sum(arr) == 0:
              return not sex
          if judge(arr):
              return sex
          if sex:
              ans = True
              for i in range(len(arr)):
                  if arr[i] != 0:
                      temp_arr = copy.deepcopy(arr)
                      temp_arr[i] -= 1
                      ans |= who_win(temp_arr, not sex)
          else:
              ans = False
              for i in range(len(arr)):
                  if arr[i] != 0:
                      temp_arr = copy.deepcopy(arr)
                      temp_arr[i] -= 1
                      ans &= who_win(temp_arr, not sex)
          return ans
      ans = []
      t = int(input())
      # man: True women: False
      for i in range(t):
          n = int(input())
          arr = [int(j) for j in input().split()]
          if n == 1:
              if arr[0] % 2 == 1:
                  ans.append("man")
              else:
                  ans.append("women")
          else:
              if who_win(arr, True):
                  ans.append("man")
              else:
                  ans.append("women")

      for word in ans:
          print(word)
    ````

<a id="I5"></a>
#### interview5. 百度2020实习生招聘-2
   - Q:
   ````python
     在浩着深邃的星空中，有若干个可以被视为质点的星球， 以及坐着飞船想要探索宇宙奥秘的度度熊。
     我们假定银河是一-个nx m的区域，顶点在(0,0)和(n,m),度度熊从最左边任意一点进入，打算穿越
     这片区域并从右边任意一点离开。在银河中分布着K个星球，每个星球以及银河的上下两个边缘都有引力，
     处于安全考虑，度度熊要离他们越远越好。试求度度熟穿越银河的路径上，距离所有星球以及上下边界的
     最小距离的最大值可以为多少?
     输入描述:
      第一行包含三个整数n,m,k(1<=n,m<=10^6, 1<=k<=6000).
      接下来k行, 每行两个整数x_i, y_i表示一个点的坐标.
      第二行n个数, 表示该排列.(1<=ai<=10^9)
     输出描述:
      一个实数表示答案, 保留4位小数
     输入:
      10 5 2
      1 1
      2 3
     输出:
      1.1180
   ````
   - A:
   ````python
   核心思想: 你要通过一个点时, 必定从其上或其下通过, 通过任意两点连线时,
   必定是从连线中点或和顶或底的中点通过.
   import math
     n, m, k = [int(i) for i in input().split()]
     arr = []
     for i in range(k):
         arr.append([float(j) for j in input().split()])
     ans = float(n) if float(n) > float(m) else float(m)
     for x, y in arr:
         distance = max(y / 2, (m - y) / 2)
         if distance < ans:
             ans = distance
     for i in range(k):
         for j in range(i + 1, k):
             x1, y1 = arr[i]
             x2, y2 = arr[j]
             distance = math.sqrt((y2 - y1)**2 + (x2 - x1)**2) / 2
             if distance < ans:
                 ans = distance
     print("%.4f"%ans)

     string = "abcdefghiihgfedcba"
     length = len(string)
     hash_map = {}

     def search_longest(char):
         max_len = 1
         ans = char
         for key, value in hash_map.items():
             if ord(key) >= ord(char) and len(value) + 1 > max_len:
                 max_len = len(value) + 1
                 ans = char + value
         return ans

     for i in range(length - 1, -1, -1):
         char = string[i]
         ans = search_longest(char)
         hash_map[char] = ans

     print(search_longest("a")[1:])
   ````

<a id="I6"></a>
#### interview6. 网易2020实习生招聘-1
   - Q:
   ````python
   厂长阿维需要根据工厂里每个员工的劳动技能等级来分配每天的工作任务。
   一个工作任务只能由劳动技能等级大于等于该任务难度的员工来完成，且
   同一天里一个员工只能对应一个工作任务, 一个工作任务只能由一个员工
   来完成. 员工i的劳动技能等级由整数Wi来表示, 工作任务i的任务难度由
   整数Ti来表示. 如何分配今天的工作任务.
   输入描述:
    第一行输入员工数N, N为整数, 且1<=N<=10^5;
    第二行输入N个员工的劳动技能等级Wi, 以空格分隔, Wi为整数, 且
    0<=Wi<=10^9;
    第三行输入N个任务的任务难度Ti, 以空格分隔, Ti为整数, 且0<=Ti<=10^9;
    第四行输入一个整数M, 且1<=M<=10^9;
   输出描述:
    输出一个整数, 该整数为所有可能的分配方式的数量除以M后所得的余数.
   输入:
    6
    1 6 3 4 5 2
    2 3 1 4 6 5
    10
   输出:
    1
   ````
   - A:
   ````python
    N = int(input())
    W = [int(i) for i in input().split()]
    T = [int(i) for i in input().split()]
    M = int(input())

    def search(W, T):
        if W == T:
            return 1
        elif W[-1] < T[-1]:
            return 0

        return (len(W[hash_map[T[-1]]:]) % M) * search(W[:-1], T[:-1]) % M

    W.sort()
    T.sort()
    hash_map = {}
    i, j = N - 1, N - 1
    if T[i] > W[j]:
        print(0)
    else:
        while i >= 0:
            while j >= 0 and W[j] >= T[i]:
                j -= 1
            if T[i] not in hash_map:
                hash_map[T[i]] = j + 1
            i -= 1
        print(search(W, T))
   ````

<a id="I7"></a>
#### interview7. 网易2020实习生招聘-2
   - Q:
   ````python
   日常收快递留下一堆快递盒占地方。 现在有N个快递盒，盒子不做翻转， 每个
   盒子有自己的长宽高数据，都以整数形式(L,W,H) 出现，将小子整理到大盒子
   里面(小盒子的长宽高都小于大盒子)。要求快递盒一个套一个(即，打开一个大
   盒子最多只能看到一个小盒子)，最多能整理多少个盒子打包到一起.
   输入:
    [[5,4,3], [5,4,5], [6,6,6]]
   输出:
    2
   输入:
    [[2,3,2],[2,2,3]]
   输出:
    1(全放不了就输出1)
   ````
   - A:
   ````python
    import copy
    def maxBoxes(boxes):
        if len(boxes) == 0:
            return 1
        boxes.sort(key=lambda x: x[0], reverse=True)
        def search(big_boxes, boxes):
            if not boxes:
                return len(big_boxes)
            ans = []
            box = boxes[0]
            for i in range(len(big_boxes)):
                if big_boxes[i][0] > box[0] and big_boxes[i][1] > box[1] and big_boxes[i][2] > box[2]:
                    temp_boxes = copy.deepcopy(big_boxes)
                    temp_boxes[i] = box
                    ans.append(search(temp_boxes, boxes[1:]))
            temp_boxes = copy.deepcopy(big_boxes)
            temp_boxes.append(box)
            ans.append(search(temp_boxes, boxes[1:]))
            return min(ans)
        num = search([], boxes)
        if num == len(boxes):
            return 1
        return search([], boxes)

    print(maxBoxes([[5,6,3],[5,4,4],[6,3,6]]))
   ````
