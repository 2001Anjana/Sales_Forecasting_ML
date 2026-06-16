# Sales_Forecasting_ML
Cinnamon Export Sales forecasting weekly demand by machine learning models
Provided with over three years of transaction-level sales data from a cinnamon export company based in Sri Lanka. The company exports a wide variety of cinnamon products to 91 countries across 19 regions worldwide. Each record in the dataset represents a single sales transaction, with the following key fields:
•	Sales Qty: The quantity of units sold in each transaction.
•	Product Code: A unique identifier for each product variant (13,700+ unique codes).
•	Order Date: The date the order was placed by the customer.
•	Invoice Date: The date the invoice was issued / shipment dispatched.
•	Region / Country: The geographic destination of the sale.
•	Sales Channel: The distribution channel — Retail, Food Service, Global, or Bulk.
•	Brand Category: The product brand segment (e.g., Retail, Food Service, Specialty & Gift, etc.).
•	Product Range: The product line grouping (e.g., Premium Grade, Bonus Grade, Exceptional Grade, etc.).
•	Sales USD: Revenue generated from the transaction in US dollars.
•	Sales KG: The weight of product sold in kilograms.
The dataset spans from early 2022 to September 2025 and contains approx- imately 60,000 transactions.


Project Objectives
1.	Forecast weekly sales quantity per product  for the next 12 weeks (3 months) using historical transaction data. You must aggregate raw transactions into weekly time series before modeling. A product is identi- fied by the first 9 characters of the Product Code. 
2.	Forecast weekly sales per product per country, drilling down into the top countries for each product to generate country-level forecasts.
3.	Handle product heterogeneity: The dataset contains both high- volume and low-volume (sparse/intermittent) products. Your approach should account for the different forecasting challenges these present.
