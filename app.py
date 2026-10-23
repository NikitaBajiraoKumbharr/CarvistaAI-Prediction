import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, hashlib, pickle, os, uuid
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="CARVISTA AI", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).parent
DB = BASE / "carvista.db"
DATA = BASE / "final_data.csv"
MODEL = BASE / "model.pkl"

# ---------- Database ----------
def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT UNIQUE, password TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT, company TEXT, car_name TEXT,
        year INTEGER, km REAL, fuel TEXT, predicted_price REAL, created_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS visits(
        session_id TEXT PRIMARY KEY, user_email TEXT, user_name TEXT,
        first_seen TEXT NOT NULL, last_seen TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS login_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT NOT NULL,
        login_at TEXT NOT NULL, session_id TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS page_views(
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        page_name TEXT NOT NULL, viewed_at TEXT NOT NULL)""")
    con.commit()
    return con

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def add_user(name,email,password):
    try:
        con=db(); con.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                              (name,email,hash_pw(password))); con.commit(); con.close(); return True
    except sqlite3.IntegrityError: return False

def check_user(email,password):
    con=db(); row=con.execute("SELECT name FROM users WHERE email=? AND password=?",
                              (email,hash_pw(password))).fetchone(); con.close(); return row[0] if row else None

def save_prediction(email, company, name, year, km, fuel, price):
    con=db(); con.execute("""INSERT INTO predictions
        (user_email,company,car_name,year,km,fuel,predicted_price,created_at)
        VALUES(?,?,?,?,?,?,?,?)""",
        (email,company,name,year,km,fuel,price,datetime.now().strftime("%Y-%m-%d %H:%M")))
    con.commit(); con.close()

def get_history(email):
    con=db(); df=pd.read_sql_query("SELECT * FROM predictions WHERE user_email=? ORDER BY id DESC",
                                   con, params=(email,)); con.close(); return df

def record_visit(session_id, email=None, name=None):
    """Count one browser session as one visitor; refreshes update last_seen."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = db()
    con.execute("""INSERT INTO visits(session_id,user_email,user_name,first_seen,last_seen)
        VALUES(?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET
        user_email=COALESCE(excluded.user_email, visits.user_email),
        user_name=COALESCE(excluded.user_name, visits.user_name), last_seen=excluded.last_seen""",
        (session_id, email, name, now, now))
    con.commit(); con.close()

def record_login(email, session_id):
    con=db()
    con.execute("INSERT INTO login_logs(user_email,login_at,session_id) VALUES(?,?,?)",
                (email, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id))
    con.commit(); con.close()

def record_page_view(session_id, page_name):
    con=db()
    con.execute("INSERT INTO page_views(session_id,page_name,viewed_at) VALUES(?,?,?)",
                (session_id, page_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit(); con.close()

def admin_credentials():
    # Configure these in .streamlit/secrets.toml or environment variables.
    username = os.getenv("CARVISTA_ADMIN_USERNAME", "")
    password = os.getenv("CARVISTA_ADMIN_PASSWORD", "")
    try:
        username = st.secrets.get("admin_username", username)
        password = st.secrets.get("admin_password", password)
    except Exception:
        pass
    return username, password

# ---------- Data/model ----------
@st.cache_data
def load_data():
    if DATA.exists():
        return pd.read_csv(DATA)
    # Demo fallback: app works even before the user's own dataset/model is added.
    rng=np.random.default_rng(7); n=500
    companies=rng.choice(["Maruti","Hyundai","Honda","Toyota","Tata","Mahindra"],n)
    names=rng.choice(["Swift","i20","City","Baleno","Creta","Nexon","Scorpio"],n)
    years=rng.integers(2012,2026,n); km=rng.integers(5000,160000,n)
    fuel=rng.choice(["Petrol","Diesel","CNG"],n)
    price=np.maximum(55000,(years-2008)*45000-km*2.0+rng.normal(0,35000,n)+
                     np.array([25000 if x=="Toyota" else 0 for x in companies]))
    return pd.DataFrame({"company":companies,"name":names,"year":years,"kms_driven":km,
                         "fuel_type":fuel,"Price":price.astype(int)})

@st.cache_resource
def load_model():
    if MODEL.exists():
        try:
            with open(MODEL,"rb") as f: return pickle.load(f)
        except Exception: pass
    return None

df=load_data()
model=load_model()

def predict(company,name,year,km,fuel):
    # First try user's trained sklearn/pipeline model.
    if model is not None:
        try:
            x=pd.DataFrame([[company,name,year,km,fuel]],
                           columns=["company","name","year","kms_driven","fuel_type"])
            return float(model.predict(x)[0])
        except Exception:
            try:
                return float(model.predict([[company,name,year,km,fuel]])[0])
            except Exception:
                pass
    # Fallback AI-style estimate based on dataset statistics.
    base=df[df["company"].astype(str).str.lower()==str(company).lower()]
    if len(base)==0: base=df
    same=base[base["name"].astype(str).str.lower()==str(name).lower()]
    if len(same)==0: same=base
    median=float(same["Price"].median())
    age=max(0,datetime.now().year-year)
    avg_km=float(same["kms_driven"].median())
    price=median*(0.91**age)*max(0.55,1-0.0000018*(km-avg_km))
    if str(fuel).lower()=="diesel": price*=1.04
    return max(30000,float(price))

# ---------- Styling ----------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:radial-gradient(circle at 10% 0%,#172554 0,#080d1d 38%,#050816 100%);color:#f8fafc}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1228,#111827);border-right:1px solid #24304d}
.hero{padding:42px 30px;border-radius:28px;background:linear-gradient(135deg,#111c3f,#172554 55%,#312e81);border:1px solid #334477;box-shadow:0 20px 60px #0006}
.hero h1{font-size:48px;margin:0;font-weight:800}.hero p{font-size:18px;color:#cbd5e1}
.card{padding:22px;border-radius:20px;background:#0f172acc;border:1px solid #263354;box-shadow:0 10px 30px #0004}
.metric{font-size:30px;font-weight:800}.muted{color:#94a3b8}
.badge{display:inline-block;padding:7px 12px;border-radius:999px;background:#312e81;color:#c4b5fd;font-weight:600}
div.stButton>button{border-radius:12px;border:0;background:linear-gradient(90deg,#4f46e5,#7c3aed);color:white;font-weight:700;padding:10px 20px}
</style>""", unsafe_allow_html=True)

db()
if "page" not in st.session_state: st.session_state.page="Home"
if "user" not in st.session_state: st.session_state.user=None
if "user_email" not in st.session_state: st.session_state.user_email=None
if "admin_logged_in" not in st.session_state: st.session_state.admin_logged_in=False
if "session_id" not in st.session_state: st.session_state.session_id=uuid.uuid4().hex
record_visit(st.session_state.session_id, st.session_state.user_email, st.session_state.user)
if st.session_state.get("last_logged_page") != st.session_state.page:
    record_page_view(st.session_state.session_id, st.session_state.page)
    st.session_state.last_logged_page=st.session_state.page

with st.sidebar:
    st.markdown("## 🚗 CARVISTA **AI**")
    st.caption("Intelligent Used Car Price Prediction")
    if st.session_state.user:
        st.success(f"Hi, {st.session_state.user}!")
    pages=["Home","Predict Price","AI Analysis","Dashboard","History","About"]
    if st.session_state.admin_logged_in:
        pages.append("Admin Dashboard")
    for p in pages:
        if st.button(p, use_container_width=True, key="nav_"+p):
            st.session_state.page=p; st.rerun()
    st.divider()
    if st.session_state.user:
        if st.button("Logout",use_container_width=True):
            st.session_state.user=None; st.session_state.user_email=None; st.session_state.page="Home"; st.rerun()
    else:
        if st.button("🔐 Login",use_container_width=True):
            st.session_state.page="Login"; st.rerun()
        if st.button("✨ Sign Up",use_container_width=True):
            st.session_state.page="Signup"; st.rerun()
    if st.session_state.admin_logged_in:
        if st.button("Admin Logout",use_container_width=True):
            st.session_state.admin_logged_in=False; st.session_state.page="Home"; st.rerun()
    else:
        if st.button("🛡️ Admin Login",use_container_width=True):
            st.session_state.page="Admin Login"; st.rerun()

page=st.session_state.page

# ---------- Login / Signup ----------
if page=="Login":
    st.markdown("<div class='hero'><span class='badge'>SECURE ACCESS</span><h1>Welcome Back 👋</h1><p>Login to access your CarVista AI dashboard.</p></div>",unsafe_allow_html=True)
    st.write("")
    c1,c2,c3=st.columns([1,1.4,1])
    with c2:
        email=st.text_input("Email")
        pw=st.text_input("Password",type="password")
        if st.button("Login",use_container_width=True):
            name=check_user(email,pw)
            if name:
                st.session_state.user=name
                st.session_state.user_email=email.strip().lower()
                record_login(st.session_state.user_email, st.session_state.session_id)
                record_visit(st.session_state.session_id, st.session_state.user_email, name)
                st.session_state.page="Predict Price"; st.rerun()
            else: st.error("Invalid email or password.")
        st.info("Demo account: create one using Sign Up.")
elif page=="Signup":
    st.markdown("<div class='hero'><span class='badge'>CREATE ACCOUNT</span><h1>Join CarVista AI ✨</h1><p>Create your account and save your prediction history.</p></div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,1.4,1])
    with c2:
        name=st.text_input("Full Name"); email=st.text_input("Email"); pw=st.text_input("Password",type="password")
        if st.button("Create Account",use_container_width=True):
            if not name or not email or len(pw)<6: st.warning("Enter all details; password must be 6+ characters.")
            elif add_user(name,email,pw): st.success("Account created! Please login."); st.session_state.page="Login"
            else: st.error("Email already registered.")

elif page=="Admin Login":
    st.markdown("<div class='hero'><span class='badge'>ADMINISTRATOR ACCESS</span><h1>Admin Login 🛡️</h1><p>Only authorized site administrators can view visitor and user activity.</p></div>",unsafe_allow_html=True)
    expected_user, expected_password = admin_credentials()
    c1,c2,c3=st.columns([1,1.4,1])
    with c2:
        admin_user=st.text_input("Admin Username", key="admin_username_input")
        admin_password=st.text_input("Admin Password", type="password", key="admin_password_input")
        if st.button("Login as Admin", use_container_width=True):
            if not expected_user or not expected_password:
                st.error("Admin credentials are not configured yet. Add admin_username and admin_password to .streamlit/secrets.toml, then restart the app.")
            elif admin_user == expected_user and admin_password == expected_password:
                st.session_state.admin_logged_in=True
                st.session_state.page="Admin Dashboard"
                st.rerun()
            else:
                st.error("Incorrect admin username or password.")
    st.caption("Set admin credentials in the private configuration file described in the setup instructions.")

elif page=="Admin Dashboard":
    if not st.session_state.admin_logged_in:
        st.error("Admin access required. Please sign in with an administrator account.")
        if st.button("Go to Admin Login"):
            st.session_state.page="Admin Login"; st.rerun()
    else:
        st.markdown("<div class='hero'><span class='badge'>PRIVATE ADMIN AREA</span><h1>Website Activity Dashboard 🛡️</h1><p>Monitor visitor sessions, registered users, successful logins and saved predictions.</p></div>",unsafe_allow_html=True)
        con=db()
        visitor_count=con.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        registered_count=con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        login_count=con.execute("SELECT COUNT(*) FROM login_logs").fetchone()[0]
        prediction_count=con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        a,b,c,d=st.columns(4)
        a.metric("Visitor Sessions", f"{visitor_count:,}")
        b.metric("Registered Users", f"{registered_count:,}")
        c.metric("Successful Logins", f"{login_count:,}")
        d.metric("Predictions Saved", f"{prediction_count:,}")
        st.info("Visitor Sessions counts browser sessions, not guaranteed unique people. Refreshes within the same Streamlit session do not add another visitor.")
        st.subheader("👥 Visitor Sessions")
        visitors=pd.read_sql_query("SELECT session_id, user_name, user_email, first_seen, last_seen FROM visits ORDER BY last_seen DESC",con)
        st.dataframe(visitors, use_container_width=True, hide_index=True)
        st.subheader("🧑‍💻 Registered Users")
        users=pd.read_sql_query("SELECT id, name, email FROM users ORDER BY id DESC",con)
        st.dataframe(users, use_container_width=True, hide_index=True)
        st.subheader("🔐 Successful Login History")
        logins=pd.read_sql_query("SELECT user_email, login_at, session_id FROM login_logs ORDER BY id DESC",con)
        st.dataframe(logins, use_container_width=True, hide_index=True)
        st.subheader("🚘 All Car Prediction History")
        predictions=pd.read_sql_query("SELECT user_email, company, car_name, year, km, fuel, predicted_price, created_at FROM predictions ORDER BY id DESC",con)
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        st.subheader("📄 Page Views")
        page_views=pd.read_sql_query("SELECT session_id, page_name, viewed_at FROM page_views ORDER BY id DESC",con)
        st.dataframe(page_views, use_container_width=True, hide_index=True)
        con.close()

elif page=="Home":
    st.markdown("""<div class='hero'><span class='badge'>AI • MACHINE LEARNING • SMART ANALYTICS</span>
    <h1>CARVISTA AI 🚗</h1><p>Predict the fair market price of a used car in seconds using machine learning and intelligent market analysis.</p></div>""",unsafe_allow_html=True)
    st.write("")
    a,b,c=st.columns(3)
    for col,title,value,desc in [(a,"Dataset Records",f"{len(df):,}","Cars available for analysis"),
                                  (b,"AI Prediction","ML","Data-driven valuation engine"),
                                  (c,"Experience","24/7","Instant prediction dashboard")]:
        with col: st.markdown(f"<div class='card'><div class='muted'>{title}</div><div class='metric'>{value}</div><p class='muted'>{desc}</p></div>",unsafe_allow_html=True)
    st.write("")
    st.subheader("✨ Why CARVISTA AI?")
    x,y,z=st.columns(3)
    with x: st.markdown("<div class='card'><h3>🤖 Smart Prediction</h3><p class='muted'>Uses machine learning features such as brand, model, year, mileage and fuel type.</p></div>",unsafe_allow_html=True)
    with y: st.markdown("<div class='card'><h3>📊 Market Insights</h3><p class='muted'>Explore price ranges, brands, fuel types and model-year trends.</p></div>",unsafe_allow_html=True)
    with z: st.markdown("<div class='card'><h3>🔐 Personal History</h3><p class='muted'>Logged-in users can save and review their previous predictions.</p></div>",unsafe_allow_html=True)
    st.write("")
    if st.button("🚀 Start Predicting"):
        st.session_state.page="Predict Price"; st.rerun()

elif page=="Predict Price":
    st.markdown("<div class='hero'><span class='badge'>MACHINE LEARNING</span><h1>Predict Your Car Price 🚘</h1><p>Enter the car details and let CARVISTA AI estimate its market value.</p></div>",unsafe_allow_html=True)
    companies=sorted(df["company"].dropna().astype(str).unique()) if "company" in df else ["Maruti"]
    company=st.selectbox("🏷️ Company",companies)
    subset=df[df["company"].astype(str)==company] if "company" in df else df
    names=sorted(subset["name"].dropna().astype(str).unique()) if "name" in subset else ["Swift"]
    name=st.selectbox("🚗 Car Model",names)
    c1,c2,c3=st.columns(3)
    with c1: year=st.number_input("📅 Manufacturing Year",min_value=2000,max_value=2026,value=2022,step=1)
    with c2: km=st.number_input("🛣️ Kilometers Driven",min_value=1000,max_value=300000,value=50000,step=1000)
    with c3: fuel=st.selectbox("⛽ Fuel Type",sorted(df["fuel_type"].dropna().astype(str).unique()) if "fuel_type" in df else ["Petrol","Diesel"])
    if st.button("🔮 Predict Price",use_container_width=True):
        price=predict(company,name,year,km,fuel)
        st.session_state["last_prediction"]=(company,name,year,km,fuel,price)
        st.markdown(f"<div class='card' style='margin-top:20px'><div class='muted'>ESTIMATED MARKET VALUE</div><div class='metric'>₹ {price:,.0f}</div><p class='muted'>Estimated using your ML model when available, with a data-driven fallback.</p></div>",unsafe_allow_html=True)
        if st.session_state.user:
            save_prediction(st.session_state.user,company,name,year,km,fuel,price)
        else: st.info("Login to save this prediction in your personal history.")

elif page=="AI Analysis":
    st.markdown("<div class='hero'><span class='badge'>AI MARKET INTELLIGENCE</span><h1>AI Analysis 📊</h1><p>Understand the factors behind used-car pricing.</p></div>",unsafe_allow_html=True)
    if "Price" in df:
        c1,c2,c3=st.columns(3)
        c1.metric("Average Price",f"₹{df.Price.mean():,.0f}")
        c2.metric("Median Price",f"₹{df.Price.median():,.0f}")
        c3.metric("Highest Price",f"₹{df.Price.max():,.0f}")
        st.subheader("Average Price by Company")
        chart=df.groupby("company")["Price"].mean().sort_values(ascending=False).head(12)
        st.bar_chart(chart)
        st.subheader("Price Distribution")
        st.line_chart(df["Price"].sort_values().reset_index(drop=True))
    else: st.warning("Price column not found.")

elif page=="Dashboard":
    st.markdown("<div class='hero'><span class='badge'>ADMIN / PROJECT DASHBOARD</span><h1>Market Dashboard 📈</h1><p>High-level analytics for your final-year project demonstration.</p></div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Cars",f"{len(df):,}")
    c2.metric("Companies",df["company"].nunique() if "company" in df else "-")
    c3.metric("Fuel Types",df["fuel_type"].nunique() if "fuel_type" in df else "-")
    c4.metric("Avg. Price",f"₹{df.Price.mean():,.0f}" if "Price" in df else "-")
    if "company" in df and "Price" in df:
        st.subheader("Top Brands by Average Price")
        st.bar_chart(df.groupby("company")["Price"].mean().sort_values(ascending=False).head(15))
    if "year" in df and "Price" in df:
        st.subheader("Year vs Price")
        st.scatter_chart(df[["year","Price"]].dropna().sample(min(500,len(df)),random_state=1))

elif page=="History":
    st.markdown("<div class='hero'><span class='badge'>YOUR ACTIVITY</span><h1>Prediction History 📜</h1><p>Saved predictions from your account.</p></div>",unsafe_allow_html=True)
    if not st.session_state.user:
        st.warning("Please login to view your history.")
    else:
        h=get_history(st.session_state.user)
        if h.empty: st.info("No predictions yet. Make your first prediction!")
        else:
            st.dataframe(h.drop(columns=["id","user_email"]),use_container_width=True,hide_index=True)

elif page=="About":
    st.markdown("<div class='hero'><span class='badge'>BCA FINAL YEAR PROJECT</span><h1>About CARVISTA AI</h1><p>An intelligent machine-learning based used car price prediction system.</p></div>",unsafe_allow_html=True)
    st.write("")
    st.markdown("""<div class='card'>
    <h2>🎓 Project Objective</h2>
    <p class='muted'>CARVISTA AI predicts the estimated resale value of used cars from historical vehicle data. It demonstrates data preprocessing, machine learning, prediction, visualization, database management and web application development.</p>
    <h2>🛠️ Technologies</h2>
    <p class='muted'>Python • Streamlit • Pandas • NumPy • Scikit-learn/XGBoost • Plotly/Charts • SQLite • HTML • CSS</p>
    <h2>📌 Modules</h2>
    <p class='muted'>Authentication, prediction engine, market analytics, recommendation-ready dataset, prediction history and responsive UI.</p>
    </div>""",unsafe_allow_html=True)

st.markdown("<br><center><span class='muted'>CARVISTA AI • BCA Final Year Project • Built with Python & Machine Learning</span></center>",unsafe_allow_html=True)
