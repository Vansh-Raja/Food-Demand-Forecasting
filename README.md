# Synthetic Data Generation

**What is it?**

Synthetic data is artificial data created using different algorithms that mirror the statistical properties of the original data but do not reveal any information regarding real people.

**Use Case:**

- Privacy
- Product testing
- Training machine learning algorithms

**Approaches:**

- **Drawing Numbers From a Distribution:**
  Create a distribution of data that follows a curve loosely based on real-world data.

- **Generative Models:**
  Involves automatically discovering and learning insights and patterns in data to output new examples matching the distribution of the real-world data.

  - **GAN (Generative Adversarial Network):**
    Treats the training process as a game between two networks - a generator and a discriminative network. The generator adjusts its model parameters to generate convincing examples to fool the discriminator, aiming to make it unable to differentiate between real and synthetic examples.

  - **VAE (Variational Autoencoder):**
    Unsupervised method where the encoder compresses the original dataset into a more compact structure. The decoder generates an output, representing the original dataset.

**Tools:**

1. **MOSTLY AI:**


  ![image](https://github.com/Vansh-Raja/Gryffindor-Internship/assets/64516886/7eb1d5a5-0658-4ee8-904f-400003044602)


2. **Gretel:**


![image](https://github.com/Vansh-Raja/Gryffindor-Internship/assets/64516886/e70efc7d-1ca8-414c-9a61-c7b9ace0fb87)


**Dataset Info:**

- **id:** Unique ID
- **week:** Number of weeks, containing 145 weeks of past data
- **center_id:** Maps with info about the center from another CSV file, like its location
- **meal_id:** Maps with meal info
- **checkout_price:** Final price including discount, taxes & delivery charges
- **base_price:** Base price of the meal
- **emailer_for_promotion:** Emailer sent for promotion of the meal
- **homepage_featured:** Meal featured at the homepage
- **num_orders:** Number of orders in that week of a particular meal type

**Fulfillment Center Info:**

- **center_id:** Unique ID for fulfillment center
- **city_code:** Unique code for the city
- **region_code:** Unique code for the region
- **center_type:** Anonymized center type
- **op_area:** Area of operation (in km^2)

**Meal Info:**

- **meal_id:** Unique ID for the meal
- **category:** Type of meal (beverages/snacks/soups….)
- **cuisine:** Meal cuisine (Indian/Italian/…)

**Problem Faced with Synthetic Data:**

Initially, as we had three different CSV files,the methods were not able to map relations properly. So, we combined all the CSVs into one and generated synthetic data, thus solving the problem.
