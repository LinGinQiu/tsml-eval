% This is a cut down version of the problem GasPricesSentiment, useful for code examples and unit tests
% The train set is reduced from 65 cases to 20 cases and the test set is reduced from 28 to 20
%
@problemName MinimalGasPrices
@timestamps false
@missing false
@univariate true
@equalLength true
@seriesLength 20
@targetlabel true
@data
3.27,3.27,2.89,2.75,2.76,2.76,2.82,2.72,2.68,2.82,2.8,2.86,2.86,2.72,2.79,2.77,2.77,2.89,2.64,2.65:-0.3745973154329336
3.2,3.2,3.13,3.14,3.11,3.1,3.06,3.03,3.03,3.23,3.24,3.26,3.27,3.27,3.13,3.11,3.24,3.22,3.22,3.1:-0.27649220292280596
2.61,2.53,2.53,2.54,2.58,2.58,2.59,2.59,2.63,2.59,2.54,2.54,2.64,2.62,2.64,2.71,2.7,2.69,2.62,2.61:-0.3359852990851952
1.88,1.96,1.96,1.91,1.96,1.99,1.9,1.97,2.01,1.93,1.71,1.76,1.94,1.94,1.96,1.92,1.97,1.97,1.88,1.88:-0.1588933546382647
2.47,2.54,2.4,2.28,2.21,2.18,2.18,2.23,2.18,2.22,2.21,2.16,2.26,2.26,2.15,2.28,2.28,2.07,2.06,2.07:-0.21406789399110351
3.42,3.1,3.25,3.25,2.72,2.8,2.74,2.89,2.92,2.95,2.95,3.36,3.54,3.58,3.58,3.43,3.43,3.43,3.13,3.13:-0.3459467332277681
2.26,2.26,2.13,2.13,2.29,2.44,2.29,2.08,2.13,2.21,2.34,2.33,2.26,2.49,2.61,2.71,2.73,2.54,2.77,2.77:-0.23691177769349164
2.9,2.91,2.91,2.9,2.9,2.89,2.89,2.86,2.76,2.81,2.8,2.75,2.79,2.79,2.73,2.82,2.82,2.78,2.75,2.82:-0.31200024393888615
3.06,2.99,2.99,2.92,2.95,2.96,2.87,2.8,2.8,2.76,2.76,2.8,2.81,2.81,2.89,2.96,3.02,2.96,2.94,2.91:-0.3456104081983749
2.81,2.77,2.71,2.89,2.82,2.82,2.86,2.65,2.57,2.49,2.45,2.63,2.73,2.77,2.76,2.68,2.88,3.24,3.01,2.99:-0.3716620670488247
3.49,3.4,3.35,3.76,6.5,6.12,11.32,23.86,8.56,4.96,3.16,2.94,2.8,2.72,2.66,2.7,2.87,2.86,2.79,2.72:-0.38911418863213976
2.92,2.92,2.89,2.91,2.87,2.76,2.76,2.83,2.85,2.76,2.76,2.85,2.84,2.93,2.93,2.83,2.79,2.71,2.77,2.77:-0.3566989751389393
2.7,2.73,2.73,2.56,2.6,2.6,2.65,2.69,2.69,2.59,2.5,2.45,2.45,2.4,2.43,2.42,2.41,2.43,2.43,2.37:-0.384325153266008
1.79,1.79,1.75,1.78,1.83,1.89,1.76,1.76,1.74,1.91,1.96,1.82,1.94,1.94,1.89,1.89,1.73,1.68,1.76,1.76:-0.12073821121683478
2.74,2.77,2.75,2.78,2.75,2.84,2.83,2.85,2.75,2.75,2.77,2.76,2.89,2.88,2.88,2.84,2.83,2.94,2.91,2.89:-0.28387368441774286
2.22,2.19,2.2,2.26,2.35,2.31,2.28,2.15,2.23,2.24,2.24,2.36,2.33,2.33,2.39,2.48,2.49,2.49,2.65,2.67:-0.2905381690424222
2.7,2.65,2.71,2.73,2.69,2.69,2.7,2.7,2.7,2.67,2.67,2.67,2.74,2.74,2.71,2.67,2.7,2.74,2.69,2.69:-0.421656129681147
2.97,3.0,3.12,3.19,3.18,3.06,3.06,3.09,3.07,2.98,2.84,2.84,2.83,2.83,3.04,3.02,3.18,3.17,3.17,3.16:-0.26659133841211974
1.78,1.92,1.9,1.86,1.81,1.68,1.8,1.73,1.63,1.69,1.78,1.93,1.9,1.84,1.74,1.7,1.61,1.56,1.6,1.66:-0.12289009959651871
4.65,4.79,4.59,4.46,4.68,4.8,4.87,4.77,5.0,5.26,5.19,5.51,5.52,5.32,5.32,5.46,5.43,5.72,5.95,6.29:-0.4409556758518402
