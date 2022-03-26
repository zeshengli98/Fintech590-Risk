from risk_management import cov,ES,psd,simulation,VaR,blackscholes

if __name__ == "__main__":
    price = simulation.PriceSimulation(1,method = "Brownian")
    print(price.data)

