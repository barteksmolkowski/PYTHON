
# dlugos = 4
# list = [0,1,0,1]
# 5 -> 101 2^2+0^2

# list = [1]

# list = [2 7 11 15]
# 2 + 7, 2 + 11, 2 + 15
# 

# O(n ^ 2)
# list[int] = ilos * rozmiar (zmiennej)
# float = 
# Python -> interpreter -> C/C++
def czyJestSuma(nums: list, target: int) -> bool:
    for i in range(len(nums)):
        
        for j in range(i + 1, len(nums)):
            if i + j == target:
                return True
    return False

# nums = [2,7,11,5], target = 12
# target = i + j
# target - i = j; gdzie to jest przejscie od 0 do len(nums): ai + ai + 1,.... 
# 12 - 2 = 10 map = {10, 5, 1} -> Map -> hash O(1) f(n) = xa + b;
# 12 - 7 = 5
# 12 - 11 = 1
# 12 - 5 = 7
def Sum(nums: list, target: int): 
    

#nums, target = input(f"taka suma nums że wynik to target: ")

print(czyJestSuma([2,5,11,15, 9], 9))

# def liczbyBin(dlugosc): # 0000
#     wszystWartosci = []

#     poczatkowa = ["0"]*dlugosc
#     print(poczatkowa)
#     i = len(poczatkowa) - 1

#     while all(poczatkowa) != "1":

#         wszystWartosci.append(poczatkowa)
#         print(f"test: początkowa: {poczatkowa} {i}")

#         if poczatkowa[i] == "0":
#             poczatkowa[i] = "1"
#             i = len(poczatkowa) - 1
            
#         else:
#             i -= 1


# print(bin(4))
# liczbyBin(4)

