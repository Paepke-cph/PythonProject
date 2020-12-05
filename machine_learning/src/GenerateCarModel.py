# Year 2000 - 2020
# Each year after 2000 + 5000kr
# Each 10k km -(5)000 kr

import random
import csv

def makeCar():
    make = 'Mercedes-Benz'
    year = random.randint(2000,2021)
    newprice = 700000
    kmperliter = 25
    km = random.randint(0,200000)
    horsepower = random.randint(50, 120)
    gearbox = random.choice([0,1])
    fueltype = 0
    doornumber = 5
    price = newprice + ((year - 2000) * random.randint(2000,2200))
    price = price - ((km / 10000) * random.randint(2000,2200))
    price = price #+ (gearbox * 20000)
    price = price + (kmperliter * 300)
    price = price + (len(make) * 1000)
    car = {'make':make, 'year':year,'newprice':newprice,'kmperliter':kmperliter,'km':km,'horsepower':horsepower,'gearbox':gearbox, 'fueltype':fueltype,'doornumber':doornumber,'price':price}
    return car

if __name__ == "__main__":
    fields = ['make','year','newprice','kmperliter','km','horsepower','gearbox','fueltype','doornumber','price']

    with open('model.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames = fields) 
        writer.writeheader()
        for i in range(1,30):
            writer.writerow(makeCar())