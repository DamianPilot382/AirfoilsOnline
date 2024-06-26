import json
import numpy as np
import pandas as pd

def lambda_handler(event, context):
    
    try:
        body = json.loads(event['body'])
        m = body['maxCamber']
        p = body['maxCamberLoc']
        t = body['maxThickness']
        n = body['numPoints']
        
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
            
        xu = np.flip(xu)
        yu = np.flip(yu)
        
        # Delete to remove the duplicate x: 0, y: 0 point
        x = np.append(np.delete(xu, -1), xl)
        y = np.append(np.delete(yu, -1), yl)
        
        datapoints = pd.DataFrame({'x': x, 'y': y})
                
        return {
            "statusCode": 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': '*'
            },
            "body": datapoints.to_json(orient='records')
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': '*'
            },
            "body": str(e)
        }
