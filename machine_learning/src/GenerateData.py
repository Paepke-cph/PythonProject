# Year 2000 - 2020
# Each year after 2000 + 5000kr
# Each 10k km -(5)000 kr

import random
import csv

def makeCar():
    make = random.choice(['Alfa Romeo','BMW','Chevrolet','Dodge','Fiat','Honda','Jaguar','Kia','Mazda','Mercedes-Benz'])
    year = random.randint(2000,2021)
    newprice = random.randint(300000,700000)
    kmperliter = random.randint(10,30)
    km = random.randint(0,200000)
    horsepower = random.randint(50, 120)
    gearbox = random.choice([0,1])
    fueltype = random.choice([0,1])
    doornumber = random.choice([3,5])
    price = newprice - ((year - 2000) * 5000)
    price = price - ((km / 10000) * 5000)
    car = {'make':make, 'year':year,'newprice':newprice,'kmperliter':kmperliter,'km':km,'horsepower':horsepower,'gearbox':gearbox, 'fueltype':fueltype,'doornumber':doornumber,'price':price}
    return car

if __name__ == "__main__":
    fields = ['make','year','newprice','kmperliter','km','horsepower','gearbox','fueltype','doornumber','price']

    with open('cars.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames = fields) 
        writer.writeheader()
        for i in range(1,50):
            writer.writerow(makeCar())