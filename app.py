import os
import sqlite3
import numpy as np
import faiss
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Updated imports
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# STEP 1: Initialize the in-house SQLite DB
# -----------------------------
DB_FILE = "inhouse.db"

def init_db():
    """Creates an in-house SQLite database and inserts customer and organization data."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create a combined table with fields from both datasets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT,
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
    
    # Check if table is empty
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        # Insert data from first document combined with relevant fields from second document
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
        ]
        
        cursor.executemany('''
            INSERT INTO customers (
                customer_id, age, gender, location, occupation, income_per_year, education,
                interests, preferences, transaction_type, category, amount_usd, purchase_date,
                payment_mode, platform, content, timestamp, sentiment_score, intent,
                industry, revenue_range, employee_count_range
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', customer_data)
    
    conn.commit()
    conn.close()
init_db()

# -----------------------------
# STEP 2: Set up banking products and FAISS vector index with OpenAI embeddings
# -----------------------------
products = [
    # Bank Accounts - Retail Banking
    {"id": 1, "name": "Savings Account", "description": "For everyday transactions and accumulating funds with competitive interest rates."},
    {"id": 2, "name": "Current Account", "description": "Designed for frequent transactions, ideal for business owners and entrepreneurs."},
    {"id": 3, "name": "Fixed Deposit Account", "description": "Enables earning higher interest rates on a lump sum amount over a fixed term."},
    {"id": 4, "name": "NRI Account", "description": "Tailored banking solution for Non-Resident Indians with special benefits."},

    # Bank Accounts - Business Banking
    {"id": 5, "name": "Business Current Account", "description": "Supports frequent business transactions with added flexibility."},
    {"id": 6, "name": "Overdraft Facility", "description": "Allows businesses to withdraw more money than available, up to a limit."},

    # Bank Cards - Credit Cards
    {"id": 7, "name": "ACE Credit Card", "description": "Cashback card with unlimited rewards, dining offers, and lounge access."},
    {"id": 8, "name": "Flipkart Shop Credit Card", "description": "Co-branded card offering cashback and discounts on Flipkart and Myntra."},
    {"id": 9, "name": "SELECT Credit Card", "description": "Premium card with Amazon vouchers, grocery discounts, and lounge visits."},
    {"id": 10, "name": "Atlas Credit Card", "description": "Super premium travel card with milestone rewards and lounge access."},
    {"id": 11, "name": "Vistara Infinite Credit Card", "description": "Travel card with business class tickets and Vistara Gold membership."},

    # Bank Cards - Debit Cards
    {"id": 12, "name": "Delights Debit Card", "description": "Offers cashback on fuel, dining, OTT subscriptions, and movie tickets."},
    {"id": 13, "name": "Priority Platinum Debit Card", "description": "High-limit card with cashback on movies and fuel surcharge waivers."},

    # Other Banking Products - Loans
    {"id": 14, "name": "Personal Loan", "description": "Flexible loan for personal needs with attractive interest rates and waived fees."},
    {"id": 15, "name": "Home Loan", "description": "Financing for home purchases with low interest and zero processing fees."},
    {"id": 16, "name": "Business Loan", "description": "Funding for business operations with flexible repayment options."},

    # Other Banking Products - Investments
    {"id": 17, "name": "Mutual Funds", "description": "Diversified investment portfolio with potential for stable returns."},
    {"id": 18, "name": "Bank Fixed Deposits", "description": "Safe, low-risk savings option with guaranteed returns."},
    {"id": 19, "name": "Public Provident Fund (PPF)", "description": "Long-term tax-saving investment with assured returns."},

    # Other Banking Products - Insurance
    {"id": 20, "name": "Term Insurance", "description": "Pure life cover providing financial protection to beneficiaries."},
    {"id": 21, "name": "Health Insurance", "description": "Covers medical expenses and hospitalization with no-claim bonuses."},
    {"id": 22, "name": "Home Insurance", "description": "Protects home structure and belongings with bundle discounts."},

    # Other Banking Products - Digital Payments
    {"id": 23, "name": "UPI (Unified Payments Interface)", "description": "Instant mobile money transfers for seamless transactions."},
    {"id": 24, "name": "Online Banking", "description": "Convenient access to accounts and transactions online."},
]

def initialize_product_embeddings():
    """Generate embeddings for product descriptions using OpenAIEmbeddings."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")

    # Initialize OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

    # Generate embeddings for product descriptions
    product_texts = [product["description"] for product in products]
    product_embeddings = embedding_model.embed_documents(product_texts)

    # Convert to numpy array for FAISS
    product_embeddings_np = np.array(product_embeddings, dtype='float32')
    dimension = product_embeddings_np.shape[1]
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(product_embeddings_np)
    
    return index, embedding_model

# Initialize embeddings and index globally
index, embedding_model = initialize_product_embeddings()

# -----------------------------
# STEP 3: Function to fetch customer details from SQLite DB
# -----------------------------
def get_customer_details(customer_id):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Enables dict-like access to rows
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# -----------------------------
# STEP 4: Function to generate recommendations using LangChain (GPT-4o)
# -----------------------------
def get_llm_recommendations(customer_data):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not found in environment variables."
    
    # Initialize LangChain's ChatOpenAI model with GPT-4o
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    
    # Build the prompt template using customer details
    prompt = ChatPromptTemplate.from_template(
"Customer details: {customer_data}. Is this customer an individual or an organization? Based on this information, what specific banking products or services might be relevant for this customer? Please list the recommendations and provide a brief explanation for each, considering whether the customer is an individual or an organization."
    )
    
    # Create a simple chain: prompt -> LLM -> output parser
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"customer_data": str(customer_data)}).strip()

# -----------------------------
# STEP 5: Function to perform vector search on banking products
# -----------------------------
def vector_search(query_text, k=5):
    """Perform vector search using embeddings generated from query text."""
    # Generate embedding for the query text
    query_embedding = embedding_model.embed_query(query_text)
    query_embedding_np = np.array([query_embedding], dtype='float32')
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding_np, k)
    print(distances)
    return [products[i] for i in indices[0]]

# -----------------------------
# STEP 6: Main Streamlit UI function
# -----------------------------
def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Banking Product Recommender", page_icon="🏦", layout="wide")

    # Title and description with custom styling
    st.title("🏦 Banking Product Recommender")
    st.markdown("""
        <style>
        .big-font {
            font-size: 18px !important;
            color: #555555;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        </style>
        <p class="big-font">Enter a Customer ID to fetch details and receive personalized banking product recommendations powered by GPT-4o.</p>
    """, unsafe_allow_html=True)

    # Input section
    st.subheader("Customer Input")
    col1, col2 = st.columns([3, 1])
    with col1:
        customer_id = st.text_input("Customer ID (e.g., 1 or 2)", placeholder="Enter a number")
    with col2:
        st.write("")  # Spacer
        submit_button = st.button("Get Recommendations")

    # Process input and display results
    if submit_button and customer_id:
        try:
            customer = get_customer_details(customer_id)
            if not customer:
                st.error(f"No customer found with ID {customer_id}")
            else:
                # Display customer details in an expander
                with st.expander("Customer Details", expanded=True):
                    st.json(customer)  # Cleaner display using JSON format

                # Get LLM recommendations using GPT-4o via LangChain
                with st.spinner("Generating recommendations with GPT-4o..."):
                    llm_query = get_llm_recommendations(customer)
                    if "API key not found" in llm_query:
                        st.error(llm_query)
                    else:
                        recommendations = vector_search(llm_query)

                        # Display LLM-generated query
                        st.subheader("AI-Generated Recommendation")
                        st.write(llm_query)

                        # Display recommended products in a card-like layout
                        st.subheader("Recommended Banking Products")
                        for prod in recommendations:
                            st.markdown(f"""
                                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                                    <h4 style="color: #4CAF50; margin: 0;">{prod['name']}</h4>
                                    <p style="margin: 5px 0 0 0;">{prod['description']}</p>
                                </div>
                            """, unsafe_allow_html=True)
        except ValueError:
            st.error("Please enter a valid numerical Customer ID.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

   
if __name__ == "__main__":
    main()