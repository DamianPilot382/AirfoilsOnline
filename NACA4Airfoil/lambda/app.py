import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lambda_handler(event, context):
    
    m = event['m']
    p = event['p']
    t = event['t']

    
    showPlot = False
    
    m = m/100
    p = p/10
    t = t/100
    
    # Define the x-coordinates of the mean camber line points\
    # and the thickness distribution
    x = np.linspace(0,1,n)
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    
    xu = np.linspace(0,1,n)
    yu = np.linspace(0,1,n)
    xl = np.linspace(0,1,n)
    yl = np.linspace(0,1,n)
    
    i = 0
    
    while i < x.size:
        if x[i] < p:
            yc = m/p**2*(2*p*x[i]-x[i]**2)
            der = 2*m/p**2*(p-x[i])
        else:
            yc = m/(1-p)**2*((1-2*p)+2*p*x[i]-x[i]**2)
            der = 2*m/(1-p)**2*(p-x[i])
            
        theta = np.arctan(der)
        
        xu[i] = x[i]-yt[i]*np.sin(theta)
        yu[i] = yc+yt[i]*np.cos(theta)
        xl[i] = x[i]+yt[i]*np.sin(theta)
        yl[i] = yc-yt[i]*np.cos(theta)
        
        i += 1
    
    if showPlot:
        plt.plot(xu, yu)
        plt.plot(xl, yl)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Airfoil')
        plt.grid(True)
        ax = plt.gca()
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 0.5])       
        plt.show()        
        
    xu = np.flip(xu)
    yu = np.flip(yu)
    
    x = np.append(xu, xl)
    y = np.append(yu, yl)
    
    datapoints = pd.DataFrame({'x': x, 'y': y})
        
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "hello world",
            "datapoints": datapoints.to_json(orient='records')
        }),
    }
