from random import uniform
class Portfolio():
    def __init__(self):
        self.cash = 0
        self.stock = {}
        self.mutual = {}
        self.transactions=[]
        self.stockprice = {}

    def addCash(self,add_amount):
        self.cash += add_amount
        self.transactions.append("You added "+str(add_amount)+" to the cash")
    
    def withdrawCash(self,wd_amount):
        self.cash -= wd_amount
        self.transactions.append("You withdrew "+str(wd_amount)+" of your cash")
       
    def buyStock(self,share,s):
        self.cash -= share*s.price
        if s.ticker in self.stock:
            self.stock[s.ticker] += share
        else:
            self.stock[s.ticker] = share
        self.stockprice[s.ticker] = s.price
        self.transactions.append("You bought "+str(share)+" of stock "+str(s.ticker)+"")
      
    def sellStock(self,ticker,share):
        self.stock[ticker] -= share
        self.cash += share*uniform(self.stockprice[ticker]*0.5,self.stockprice[ticker]*1.5)
        self.transactions.append("You sold "+str(share)+" of stock "+ticker+"")
        
    def buyMutualFund(self,share,mf):
        self.cash -= share
        if mf.ticker in self.mutual:
            self.mutual[mf.ticker] += share
        else:
            self.mutual[mf.ticker] = share
        self.transactions.append("You bought "+str(share)+" of mutual fund "+str(mf.ticker)+"")
     
    def sellMutualFund(self,mf,share):
        self.mutual[mf] -= share
        self.cash += share*uniform(0.9,1.2)
        self.transactions.append("You sold "+str(share)+" of mutual fund "+mf+"")
       
    def history(self):
        return print(*self.transactions, sep='\n')
    def __str__(self):
        return "cash: " + str(self.cash) +"\n"+ "stock: " + str(self.stock) + "\n"+ "mutual funds: " + str(self.mutual)
    def __repr__(self):
        return self.__str__()
    
class Stock(Portfolio):
    def __init__(self, price, ticker):
        self.price = price
        self.ticker = ticker
        
class MutualFund(Portfolio):
    def __init__(self, ticker):
        self.ticker = ticker


class Bonds(Portfolio):
    def __init__(self, ticker):
        self.ticker = ticker
        
"""       
portfolio = Portfolio() #Creates a new portfolio
portfolio.addCash(300.50) #Adds cash to the portfolio
s = Stock(20, "HFH") #Create Stock with price 20 and symbol "HFH"
portfolio.buyStock(5, s) #Buys 5 shares of stock s
mf1 = MutualFund("BRT") #Create MF with symbol "BRT"
mf2 = MutualFund("GHT") #Create MF with symbol "GHT"
portfolio.buyMutualFund(10.3, mf1) #Buys 10.3 shares of "BRT"
portfolio.buyMutualFund(2, mf2) #Buys 2 shares of "GHT"
print(portfolio)
portfolio.sellMutualFund("BRT", 3) #Sells 3 shares of BRT
portfolio.sellStock("HFH", 1) #Sells 1 share of HFH
portfolio.withdrawCash(50) #Removes $50
portfolio.history() #Prints a list of all transactions ordered by time
"""
