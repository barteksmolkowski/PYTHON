class mathFunc:
    def __init__(self, number):
        from sympy import isprime
        self.isprime = isprime
        
        self.number = number

    def is_prime(self) -> bool:
        return self.isprime(self.number)

    def is_semiprime(self) -> bool:
        count = 0
        num = self.number
        for i in range(2, num + 1):
            while num % i == 0 and self.is_prime(i):
                num //= i
                count += 1
                if count > 2:
                    return False
                
        return count == 2

    def is_halftwin_first(self) -> bool:
        if self.is_prime(self.number) and self.is_prime(self.number + 2):
            return True
        
        return False

    def len_period(self, bigger: int) -> bool:
        num = self.number
        while num % 2 == 0:
            num //= 2
        while num % 5 == 0:
            num //= 5

        if num == 1:
            return 0

        k = 1
        mod = 10 % num
        while mod != 1:
            mod = (mod * 10) % num
            k += 1
        
        if k >= bigger:
            return True
        else:
            return False
