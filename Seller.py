class Seller:
    def __init__(self, capital, fixed_price, is_vacant):
        self.__capital = 0
        self.__fixed_price = 0
        self.__is_vacant = 0

    def get_capital(self):
        return self.__capital

    def set_capital(self, capital):
        self.__capital = capital

    def get_fixed_price(self):
        return self.__fixed_price

    def set_fixed_price(self, fixed_price):
        self.__fixed_price = fixed_price

    def get_vacant(self):
        return self.__is_vacant

    def set_vacant(self, is_vacant):
        self.__is_vacant = is_vacant
