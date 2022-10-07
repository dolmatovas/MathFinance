# MathFinance
My notebooks on Mathematical finance

Ongoing projects:

1. [Asian options pricing in BS model](https://github.com/dolmatovas/MathFinance/blob/main/Notebooks/Asian%20options/Asian%20option%20pricing.ipynb)
In this project I'm considering different approaches for asian option pricing, such as Monte carlo simulation, variance reduction techniques(moment matching, control variate), lognormal distribution approximation and conditioning on final spot price. I'm also going to cover different finite differences approaches and Laplace transformation technuqie.

2. [Heston model, fourier analysis and finite diferences](https://github.com/dolmatovas/MathFinance/tree/main/Notebooks/Heston%20FD)
In this project I wrote a code for pricing vanilla options in Heston Model, based on exact formula and numerical solution of corresponding Heston PDE. For solving PDE I used alternating direction scheme, which is numerically stable.


3. [Double Heston model calibration](https://github.com/dolmatovas/MathFinance/blob/main/Notebooks/Heston%20Calibration/Double%20Heston%20Calibration.ipynb)
In this project I used non-linear least squares to calibrate double Heston model. Since we have closed form solution in Heston model, we can get a closed form solution for the derivatives of option price with respect to model parameters. This allows us to effectivly calibrate this parameters.


4. [BTC options price surface calibration](https://github.com/dolmatovas/MathFinance/blob/main/CryptoOptionsCalibration/Calibration.ipynb)
In this project I calibrated Heston and Double Heston models to market data of BTC option prices. Since only OTM options are liquid enough, I used them to calibrate model and then I used calibrated parameters in order to extrapolate option curve to ITM regions.