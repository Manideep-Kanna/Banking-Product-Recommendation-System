import os
import sqlite3
import numpy as np
from langchain.vectorstores import FAISS
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List, Optional

# -----------------------------
# Configuration
# -----------------------------
DB_FILE = "inhouse.db"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Optionally, if a library expects the API key as an environment variable:
# import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# STEP 1: Initialize SQLite Database
# -----------------------------
def init_db():
    """Initialize SQLite database with customer and organization data."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                location TEXT,
                occupation TEXT,
                income_per_year INTEGER,
                education TEXT,
                interests TEXT,
                preferences TEXT,
                transaction_type TEXT,
                category TEXT,
                amount_usd INTEGER,
                purchase_date TEXT,
                payment_mode TEXT,
                platform TEXT,
                content TEXT,
                timestamp TEXT,
                sentiment_score REAL,
                intent TEXT,
                industry TEXT,
                revenue_range TEXT,
                employee_count_range TEXT
            )
        ''')

        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] == 0:
            customer_data = [
                            # Customer 1
                            ('CUST72168', 36, 'M', 'New York', 'Retired with Pension + 401(k)', 243319, 'Under-Graduate',
                                'Flights, Hotels, Adventure Activities, Cameras', 
                                'Wealth Management, Digital Business Financing',
                                'Loan EMI', 'New York to Tokyo', 353121, '2023-06-28', 'ACH Debit',
                                'Facebook', 'Random post about Fashion Focus', '2024-05-07 15:27:00', 0.55,
                                'Tech Innovation Interest', None, None, None),

                            # Customer 2
                            ('CUST95575', 60, 'F', 'Boston', 'Wealth Manager', 210453, 'Master’s',
                                'Finance Investments, Real Estate, Business Networking',
                                'Travel Credit Cards, Digital Banking, International Insurance',
                                'Flight Booking', 'New York to Tokyo', 2899837, '2023-12-03', 'Debit Card',
                                'Reddit', 'Random post about Audience Engagement', '2025-02-27 05:09:00', 0.82,
                                'Subscription Change', 'Financial Services and Wealth Management', '50M-80M', '1000-2500'),

                            # Customer 3
                            ('CUST85859', 36, 'M', 'San Francisco', 'Marketing Manager', 227020, 'PhD',
                                'Online Shopping, Food Delivery, Travel',
                                'BNPL, Crypto, Subscription Services',
                                'Flight Booking', 'Gucci', 763964, '2023-07-16', 'Affirm',
                                'Instagram', 'Random post about Sales and Expansion', '2024-04-06 16:28:00', -0.51,
                                'Supply Chain Optimization', None, None, None),

                            # Customer 4
                            ('CUST92666', 27, 'M', 'Los Angeles', 'HR Manager', 68471, 'PhD',
                                'Gaming, Tech Gadgets, Streaming Subscriptions',
                                'Private Banking, Investment Portfolio, Tax Advisory',
                                'Technology Investment', 'Branding and Social Media Ads', 3329765, '2024-04-28', 'Auto Debit',
                                'Facebook', 'Random post about Fashion Interest', '2024-06-15 09:34:00', 0.31,
                                'Fashion Interest', None, None, None),

                            # Customer 5
                            ('CUST43279', 48, 'M', 'Chicago', 'Wealth Manager', 66385, 'PhD',
                                'Budget Shopping, Dining, Mortgage Payments',
                                'Home Loan, Retirement Saving, ETFs',
                                'Expansion Loan', 'AI-Powered E-commerce Platform', 4902656, '2024-03-26', 'Credit Card',
                                'Reddit', 'Random post about Fashion Interest', '2024-08-08 07:05:00', -0.62,
                                'Cost & Financing Concern', 'Financial Services and Wealth Management', '50M-80M', '1000-2500'),
                            ('ORG_US_33554', None, None, None, None, None, None,
                            'Supply Chain Financing, Inventory Loans', 'Digital Innovation, Limited Edition Collections',
                            None, None, None, None, None, None, None, None, None, None,
                            'Financial Services and Wealth Management', '50M-80M', '1000-2500'),
                            ('ORG_US_09428', None, None, None, None, None, None,
                            'Business Loans, Sponsorship Financing', 'R&D, Ethical Sourcing',
                            None, None, None, None, None, None, None, None, None, None,
                            'Fashion And Clothing', '50M-80M', '800-1500'),
                            ('ORG_US_00200', None, None, None, None, None, None,
                            'Business Loans, Payment Processing', 'Market Research, Alternative Investments',
                            None, None, None, None, None, None, None, None, None, None,
                            'Agriculture and Organic Food Production', '500M-700M', '800-1500'),
                            ('ORG_US_12448', None, None, None, None, None, None,
                            'Agriculture Loans, Supply Chain Financing', 'R&D, Ethical Sourcing',
                            None, None, None, None, None, None, None, None, None, None,
                            'IT Services and Software Development', '150M-200M', '800-1500'),
                            ('ORG_US_39490', None, None, None, None, None, None,
                            'Agriculture Loans, Supply Chain Financing', 'R&D, Ethical Sourcing',
                            None, None, None, None, None, None, None, None, None, None,
                            'Luxury Fashion and Apparel', '80M-120M', '1000-2500'),
                            ('CUST99999', 19, 'F', 'Rural Alaska', 'Wildlife Photographer', 15000, 'High School',
                            'Wildlife Conservation, Bird Watching, Nature Photography',
                            'Eco-Friendly Products, Sustainable Living',
                            'Charity Donation', 'Wildlife Preservation Fund', 50, '2024-10-15', 'Cash',
                            'Twitter', 'Post about saving endangered species', '2024-10-16 08:00:00', -0.4,
                            'Environmental Advocacy', None, None, None)
                            ]
        
            cursor.executemany('INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', customer_data)
            conn.commit()

init_db()

# -----------------------------
# STEP 2: Banking Products and FAISS Index
# -----------------------------
PRODUCTS = [
    {"id": 3, "name": "Fixed Deposit Account", "description": "Enables earning higher interest rates on a lump sum amount over a fixed term."},
    {"id": 4, "name": "NRI Account", "description": "Tailored banking solution for Non-Resident Indians with special benefits."},
    {"id": 5, "name": "Business Current Account", "description": "Supports frequent business transactions with added flexibility."},
    {"id": 6, "name": "Overdraft Facility", "description": "Allows businesses to withdraw more money than available, up to a limit."},
    {"id": 7, "name": "ACE Credit Card", "description": "Cashback card with unlimited rewards, dining offers, and lounge access."},
    {"id": 10, "name": "Atlas Credit Card", "description": "Super premium travel card with milestone rewards and lounge access."},
    {"id": 14, "name": "Personal Loan", "description": "Flexible loan for personal needs with attractive interest rates and waived fees."},
    {"id": 15, "name": "Home Loan", "description": "Financing for home purchases with low interest and zero processing fees."},
    {"id": 16, "name": "Business Loan", "description": "Funding for business operations with flexible repayment options."},
    {"id": 17, "name": "Mutual Funds", "description": "Diversified investment portfolio with potential for stable returns."},
    {"id": 18, "name": "Bank Fixed Deposits", "description": "Safe, low-risk savings option with guaranteed returns."},
    {"id": 19, "name": "Public Provident Fund (PPF)", "description": "Long-term tax-saving investment with assured returns."},
    {"id": 20, "name": "Term Insurance", "description": "Pure life cover providing financial protection to beneficiaries."},
    {"id": 21, "name": "Health Insurance", "description": "Covers medical expenses and hospitalization with no-claim bonuses."},
    {"id": 23, "name": "UPI (Unified Payments Interface)", "description": "Instant mobile money transfers for seamless transactions."},
    {"id": 24, "name": "Online Banking", "description": "Convenient access to accounts and transactions online."},
    {"id": 25, "name": "Health Insurance", "description": "Covers medical expenses, hospitalization for vehicle owners, third-party coverage, comprehensive, and add-on covers with no-claim bonuses."},
    {"id": 26, "name": "Home Insurance", "description": "Covers home structure and contents from damages, theft, or loss with safeguards proper for homeowners."},
    {"id": 27, "name": "Fire Insurance", "description": "Covers fire damages and associated claim depends on fire damages & losses including domestic, international, individual, student, or senior citizen coverage."},
    {"id": 28, "name": "Travel Insurance", "description": "Covers financial losses due to trip cancellations, medical coverage for travel emergencies, and safeguards against travel-based policies."},
    {"id": 29, "name": "Term Life Insurance", "description": "Basic term insurance with death benefit providing financial protection to the nominee up to 85 years as per policy tenure."},
    {"id": 30, "name": "Whole Life Insurance", "description": "Offers lifelong coverage with a savings component combining insurance and saving maturity benefits or death benefits claimable over time."},
    {"id": 31, "name": "Endowment Plans", "description": "Combines life coverage with savings over time providing cash value accumulation over maturity."},
    {"id": 32, "name": "Unit-Linked Insurance Plans", "description": "Combines investment in market-linked funds with insurance coverage providing market-linked returns dependent on market performance and savings."},
    {"id": 33, "name": "Child Plans", "description": "Savings and insurance for child’s future financial security with market-linked returns to ensure future financial independence."},
    {"id": 34, "name": "Pension Plans", "description": "Retirement planning with steady post-retirement income in the form of pensions or annuities with varied payout options."},
]

PRODUCT_KEYWORDS = {
    "Fixed Deposit Account": ["savings", "interest rates", "fixed term", "deposit"],
    "NRI Account": ["non-resident", "indian", "nri", "international"],
    "Business Current Account": ["business", "transactions", "current account"],
    "Overdraft Facility": ["business", "overdraft", "withdraw", "credit"],
    "ACE Credit Card": ["cashback", "rewards", "dining", "lounge", "credit card"],
    "Atlas Credit Card": ["travel", "premium", "lounge", "credit card", "milestone rewards"],
    "Personal Loan": ["loan", "personal", "flexible", "financing"],
    "Home Loan": ["home", "mortgage", "financing", "loan"],
    "Business Loan": ["business", "loan", "funding", "operations"],
    "Mutual Funds": ["investment", "portfolio", "returns", "mutual funds"],
    "Bank Fixed Deposits": ["savings", "fixed deposit", "guaranteed returns", "low-risk"],
    "Public Provident Fund (PPF)": ["tax-saving", "investment", "long-term", "ppf"],
    "Term Insurance": ["life cover", "financial protection", "term insurance"],
    "Health Insurance": ["medical", "hospitalization", "health", "insurance"],
    "UPI (Unified Payments Interface)": ["mobile", "transfers", "upi", "payments"],
    "Online Banking": ["online", "digital", "banking", "transactions"],
    "Home Insurance": ["home", "damages", "theft", "insurance"],
    "Fire Insurance": ["fire", "damages", "insurance"],
    "Travel Insurance": ["travel", "trip", "emergencies", "insurance", "cancellations"],
    "Term Life Insurance": ["life insurance", "death benefit", "term"],
    "Whole Life Insurance": ["lifelong", "savings", "insurance", "whole life"],
    "Endowment Plans": ["life coverage", "savings", "endowment"],
    "Unit-Linked Insurance Plans": ["investment", "insurance", "market-linked", "ulip"],
    "Child Plans": ["child", "future", "savings", "education"],
    "Pension Plans": ["retirement", "pension", "income", "annuities"],
}

def initialize_product_vector_store() -> tuple[FAISS, OpenAIEmbeddings]:
    """Initialize a LangChain FAISS vector store with product embeddings."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment variables.")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    
    # Prepare product texts and metadata
    product_texts = [product["description"] for product in PRODUCTS]
    product_metadata = [{"id": product["id"], "name": product["name"], "description": product["description"]} for product in PRODUCTS]
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(
        texts=product_texts,
        embedding=embedding_model,
        metadatas=product_metadata
    )
    
    return vector_store, embedding_model

# Initialize the vector store and embedding model
VECTOR_STORE, EMBEDDING_MODEL = initialize_product_vector_store()

# -----------------------------
# STEP 3: Fetch Customer Details
# -----------------------------
def get_customer_details(customer_id: str) -> Optional[Dict]:
    """Fetch customer details from SQLite database."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def generate_similarity_query(customer_data: Dict) -> str:
    """
    Generate a similarity search query for a customer (individual or organization) based on all fields from the customers table,
    with business rules applied to sentiment_score.
    
    Args:
        customer_data (dict): Customer data with fields matching the customers table.
    
    Returns:
        str: Formatted query string for vector database similarity search.
    """
    query_parts = []

    # Check if this is an organization (no age, gender, or occupation)
    is_organization = not (customer_data.get('age') or customer_data.get('gender') or customer_data.get('occupation'))

    if is_organization:
        # Organizational query
        query_parts.append("I represent an organization")
        if customer_data.get('industry'):
            query_parts.append(f"in the {customer_data['industry']} industry")
        if customer_data.get('revenue_range'):
            query_parts.append(f"with a revenue range of {customer_data['revenue_range']}")
        if customer_data.get('employee_count_range'):
            query_parts.append(f"and an employee count of {customer_data['employee_count_range']}")
    else:
        # Individual query
        if customer_data.get('age') and customer_data.get('gender') and customer_data.get('occupation') and customer_data.get('location'):
            query_parts.append(
                f"I’m a {customer_data['age']}-year-old {customer_data['gender']} {customer_data['occupation']} from {customer_data['location']}"
            )
        if customer_data.get('income_per_year'):
            query_parts.append(f"with an income of {customer_data['income_per_year']} per year")
        if customer_data.get('education'):
            query_parts.append(f"and a {customer_data['education']} education")

    # Interests and preferences (common to both)
    if customer_data.get('interests'):
        query_parts.append(f"My interests are {customer_data['interests']}")
    if customer_data.get('preferences'):
        query_parts.append(f"and I prefer {customer_data['preferences']}")

    # Transaction details (common to both)
    if customer_data.get('transaction_type') and customer_data.get('category'):
        query_parts.append(
            f"I recently made a {customer_data['transaction_type']} transaction for {customer_data['category']}"
        )
        if customer_data.get('amount_usd'):
            query_parts.append(f"costing {customer_data['amount_usd']} USD")
        if customer_data.get('purchase_date'):
            query_parts.append(f"on {customer_data['purchase_date']}")
        if customer_data.get('payment_mode'):
            query_parts.append(f"via {customer_data['payment_mode']}")
        if customer_data.get('platform'):
            query_parts.append(f"on {customer_data['platform']}")

    # Content and behavioral data (common to both)
    if customer_data.get('content'):
        content_str = (
            f"The content related to this was '{customer_data['content']}'"
            + (f" (timestamp: {customer_data['timestamp']})" if customer_data.get('timestamp') else "")
        )
        query_parts.append(content_str)

    # Sentiment score business rules (common to both)
    sentiment_score = customer_data.get('sentiment_score')
    if sentiment_score is not None:  # Only add if sentiment_score exists
        if sentiment_score <= -0.3:
            query_parts.append("I’m feeling concerned about costs and need affordable or supportive solutions")
        elif -0.3 < sentiment_score <= 0.3:
            query_parts.append("I’m looking for practical and straightforward options")
        elif sentiment_score > 0.3:
            query_parts.append("I’m optimistic and seeking premium or growth-oriented solutions")

    if customer_data.get('intent'):
        query_parts.append(f"and my intent is {customer_data['intent']}")

    # Combine all parts into a single query
    query = ". ".join(query_parts) + ". What banking products match my profile, needs, and behavior?"
    return query
# -----------------------------
# STEP 4: Vector Search with Sentiment
# -----------------------------
# Fix in vector_search function

def vector_search(customer_data: Dict, k: int = 10) -> List[Dict]:
    """Perform vector search using LangChain FAISS, incorporating all customer data and sentiment."""
    if not customer_data:
        raise ValueError("Customer data is required for vector search.")
    
    # Generate the query using the template
    query_text = generate_similarity_query(customer_data)
    
    # Perform similarity search with scores
    results = VECTOR_STORE.similarity_search_with_score(query_text, k=k)
    # Convert L2 distances to similarity scores
    similarities = [100 * (1 / (1 + distance)) for _, distance in results]
    
    # Debug output: print results and similarity scores
    print("Search Results:")
    for (doc, distance), similarity in zip(results, similarities):
        print(f"Product: {doc.metadata['name']}, L2 Distance: {distance:.4f}, Similarity Score: {similarity:.2f}%")
    
    # Filter products with similarity score > 50%
    filtered_products = []
    for (doc, _), similarity in zip(results, similarities):
        if similarity > 42:  # Lowered threshold to 45%
            product = {
                "id": doc.metadata["id"],
                "name": doc.metadata["name"],
                "description": doc.metadata["description"]
            }
            filtered_products.append(product)
    
    return filtered_products
# -----------------------------
# STEP 5: Generate Recommendations with LLM
# -----------------------------
def get_llm_recommendations(customer_data: Dict, retrieved_products: List[Dict]) -> str:
    """Generate recommendations using GPT-4o with retrieved products and sentiment."""
    if not OPENAI_API_KEY:
        return "OpenAI API key not found in environment variables."
    
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    product_list = "\n".join([f"- {p['name']}: {p['description']}" for p in retrieved_products])
    
    # Interpret sentiment for the prompt
    sentiment_score = customer_data.get('sentiment_score', 0.0)
    if sentiment_score <= -0.3:
        sentiment_context = "The customer has a negative sentiment (score: {sentiment_score}), indicating potential dissatisfaction or concerns."
    elif sentiment_score >= 0.3:
        sentiment_context = "The customer has a positive sentiment (score: {sentiment_score}), indicating satisfaction or optimism."
    else:
        sentiment_context = "The customer has a neutral sentiment (score: {sentiment_score})."
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert banking product recommender. Your task is to analyze customer details and recommend 3-5 banking products from a provided list that best match the customer's profile, needs, and preferences. The customer's sentiment has already influenced the initial product retrieval, so focus on refining these recommendations. Follow these steps:

    1. **Customer Analysis**:
       - Customer details: {customer_data}
       - Identify if the customer is an **individual** or an **organization**:
         - **Individual**: Look for personal attributes like age, gender, occupation, income_per_year, education, location.
         - **Organization**: Look for organizational attributes like industry, revenue_range, employee_count_range (personal attributes will be absent).
       - Note key interests, preferences, transaction history, intent, and any sentiment-related phrasing (e.g., "concerned about costs" or "seeking premium solutions") already embedded in the profile.

    2. **Product Recommendation**:
       - Available products (pre-filtered based on customer profile and sentiment):
         {product_list}
       - Select 3-5 products that align with the customer’s attributes, needs, and preferences.
       - For each recommendation, provide:
         - **Product Name**: The exact name from the list.
         - **Reason**: A 1-2 sentence explanation tying the product to specific customer details (e.g., interests, preferences, transaction type, industry).

    3. **Output Format**:
       - Use this structure for clarity:
       """
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "customer_data": str(customer_data),
        "sentiment_context": sentiment_context.format(sentiment_score=sentiment_score),
        "product_list": product_list
    }).strip()
    
    
def document_customer_info(customer: Dict) -> None:
    """Document customer information in a single text file for all customers with no relevant products."""
    filename = "unmatched_customers.txt"
    
    # Use 'a' mode to append to the file (creates the file if it doesn't exist)
    with open(filename, "a") as f:
        f.write(f"Customer Information for {customer.get('customer_id', 'unknown')}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in customer.items():
            f.write(f"{key}: {value}\n")
        f.write("\nReason for Documentation:\n")
        f.write("No banking products found with a relevance score above 50%.\n")
        f.write("\n" + "=" * 50 + "\n\n")  # Separator between customers

# -----------------------------
# STEP 6: Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Banking Product Recommender", page_icon="🏦", layout="wide")
    st.title("🏦 Banking Product Recommender")
    st.markdown("""
        <style>
        .big-font {font-size: 18px; color: #555555;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px;}
        </style>
        <p class="big-font">Enter a Customer ID (e.g., CUST72168) to get personalized banking product recommendations.</p>
    """, unsafe_allow_html=True)

    with st.form(key="customer_form"):
        customer_id = st.text_input("Customer ID", placeholder="e.g., CUST72168")
        submit_button = st.form_submit_button(label="Get Recommendations")

    if submit_button and customer_id:
        with st.spinner("Fetching customer data..."):
            customer = get_customer_details(customer_id)
            if not customer:
                st.error(f"No customer found with ID '{customer_id}'. Please check the ID and try again.")
            else:
                st.subheader("Customer Details")
                st.json(customer)

                with st.spinner("Retrieving relevant products..."):
                    retrieved_products = vector_search(customer, k=10)

                    if retrieved_products:  # If there are products above the 50% threshold
                        with st.spinner("Generating recommendations with GPT-4o..."):
                            llm_response = get_llm_recommendations(customer, retrieved_products)
                            if "API key not found" in llm_response:
                                st.error(llm_response)
                            else:
                                st.subheader("AI-Generated Recommendations")
                                st.write(llm_response)

                                st.subheader("Recommended Banking Products")
                                for prod in retrieved_products:
                                    if prod["name"] in llm_response:
                                        st.markdown(f"""
                                            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                                                <h4 style="color: #4CAF50; margin: 0;">{prod['name']}</h4>
                                                <p style="margin: 5px 0 0 0;">{prod['description']}</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                    else:  # If no products meet the threshold, document customer info
                        st.warning("No products found with a relevance score above 50%. Documenting customer information instead.")
                        document_customer_info(customer)
                        st.success(f"Customer information for {customer_id} has been appended to 'unmatched_customers.txt'.")

if __name__ == "__main__":
    main()