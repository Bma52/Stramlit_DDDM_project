
#Import necessary packages.
import pandas as pd 
import streamlit as st 
import hydralit_components as hc
import time
import hydralit as hy
import sklearn 
import plotly.express as px
import functools
import pickle 
import seaborn as sns
from dash import html , dcc
from traitlets import Float



# Set the layout of the page initially to wide and collapsed sidebar menu for a nicer overview.
st.set_page_config(
       page_title="Telco Churn Analysis & Prediction App",
       page_icon="🧊",
       layout="wide",
       initial_sidebar_state="collapsed",
       menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

#with hc.HyLoader('Loading ... Please Wait',hc.Loaders.pulse_bars):
#        time.sleep(2)

st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> Telco Customer Churn Prediction and Analysis App</font></H1></div>', unsafe_allow_html=True)


# A function that calls the Logistic regression model, fit new data, and predict the churn label. Fit batch of customer data. 
def churn_predictions_batch(df: pd.DataFrame) -> pd.DataFrame:
    
    file = open("lr_model_pkl",'rb')
    model = pickle.load(file)
    
    X_features = df[['gender', 'SeniorCitizen', 'Partner', 'Tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 
        'Contract', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']]

    Y_label = model.predict(X_features)
    df["Churn"] = Y_label
    

    return df


# A function taht calls the stored logistic model and pass the input customer information, to predict whether this customer will Churn or no. 
def input_churn_predictions(gender: int, SeniorCitizen: int, Partner: int, Tenure: str, PhoneService: int, 
              MultipleLines: int, InternetService: int, Contract: str, PaymentMethod: str, MonthlyCharges: int, TotalCharges: int) -> bool:
    
    file = open("lr_model_pkl",'rb')
    model = pickle.load(file)
    gender_list = [gender]
    SeniorCitizen_list = [SeniorCitizen]
    Partner_list = [Partner]
    Tenure_list = [Tenure]
    PhoneService_list = [PhoneService]
    MultipleLines_list = [MultipleLines]
    InternetService_list = [InternetService]
    Contract_list = [Contract]
    PaymentMethod_list = [PaymentMethod]
    MonthlyCharges_list = [MonthlyCharges]
    TotalCharges_list = [TotalCharges]
    df = pd.DataFrame(list(zip(gender_list, SeniorCitizen_list, Partner_list, Tenure_list, PhoneService_list
    , MultipleLines_list, InternetService_list, Contract_list, PaymentMethod_list, MonthlyCharges_list, TotalCharges_list)),
     columns=['gender', 'SeniorCitizen', 'Partner', 'Tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 
        'Contract', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges'])
    df['gender'] = df['gender'].map({"Female": 1, "Male": 0}) 
    df['PaymentMethod'] = df['PaymentMethod'].map({"Bank Transfer":0 , "Credit Card": 1, "Electronic Ckeck":2 , "Mailed Check": 3}) 
    df['Contract'] = df['Contract'].map({"Month-to-Month":0, "One Year":1, "Two years": 2}) 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['InternetService'] = df['InternetService'].map({"No":0 , "DSL":1 , "Fiber Optic":2 }) 
    df['Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce')


    X_features = df
    Y_label = model.predict(X_features)
    return Y_label




# A function to convert the datarame to csv to use it in the downlaod button. 
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')




# A function to convert some coded/binary columns into categorical to better understand them in visuals. 
def manipulate_data(df: pd.DataFrame) -> pd.DataFrame:

    df['gender'] = df['gender'].map({1: "Female", 0: "Male"}) 
    df['PaymentMethod'] = df['PaymentMethod'].map({0: "Bank Transfer", 1: "Credit Card", 2: "Electronic Ckeck", 3:"Mailed Check"}) 
    df['Contract'] = df['Contract'].map({0: "Month-to-Month", 1: "One Year", 2: "Two years"}) 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['InternetService'] = df['InternetService'].map({0: "No", 1: "DSL", 2: "Fiber Optic"}) 

    return df




# a function to highlight the cells in the table having Churn = 1 
def highlight_churn(val):
    
     color = 'green' if val==0 else 'red'
     return f'background-color: {color}'





# Menu Navigation Bar construction
menu_data = [
    {'icon': "far fa-copy", 'label':"Input Data for Churn Prediction"},
    {'icon': "far fa-copy", 'label':"Batch Prediction"},#no tooltip message
    {'icon': "fa-thin fa-address-book",'label':"Customers Profiling", 'submenu':[{'label':"Churn", 'icon': "fa fa-meh"},{'icon':'fas fa-smile','label':"Not Churn",}]},
]

over_theme = {'txc_inactive': '#FFFFFF', 'menu_background':'Green'}
menu_id = hc.nav_bar(
       menu_definition=menu_data,
       override_theme=over_theme,
       home_name='Analysis Dashboard',
       hide_streamlit_markers=False, 
       sticky_nav=True, #at the top or not
       sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)




user_s_params = {'session_id': 50}
# Main Hydra App that will include all other apps when running. 
app = hy.HydraApp(nav_horizontal=True,  layout='wide', navbar_mode='pinned',use_navbar=True, session_params=user_s_params)
st.session_state = 0




# Home Tab is teh analysis dashboard main page. This page visualizes the customer base of Telco company. 
# The analsysis aims to study Churn with other dimensions to see what are teh main factors affecting the Churn decision for a customer.
#  This page include a sidebar with filters taht will dynamically change the visuals accordingly in this dashboard. 

@app.addapp()
def Home():
  st.session_state = 0
  company_data = open("/Users/bothainaa/Desktop/Streamlit Churn App/Train_telco.csv", "r")
  df = pd.read_csv(company_data)
  df = manipulate_data(df)

  # Sidebar section 
  st.sidebar.subheader("Filter by Gender")

  gender = ["All", "Female", "Male"]
    
  gender_selections = st.sidebar.radio(
        "Select Gender to filter", gender)


  payment_method = ["All", "Bank Transfer", "Credit Card", "Electronic Ckeck", "Mailed Check"]     
  st.sidebar.subheader("Filter by Payment Method") 
  payment_selections = st.sidebar.selectbox(
        "Select Payment Method to filter by", payment_method)

  if gender_selections == "All":
        df = df
  else: 
        if gender_selections == "Female":
            df = df.loc[df["gender"] == "Female"]
        else:
            df = df.loc[df["gender"] == "Male"]


  if payment_selections == "All":
        df = df
  else: 
        df = df.loc[df["PaymentMethod"] == payment_selections]


  # KPIs section represented as info cards 
  churn_rate = len(df.loc[df["Churn"] == 1])*100/len(df.customerID.unique())
  churns = len(df.loc[df["Churn"] == 1])
  not_churns = len(df.loc[df["Churn"] == 0])

  st.info("Welcome to the Telco customer base analysis dahsboard. Mainly we are Analyzing Churn and the factors that most probably affect customer churn decision.")
  st.subheader("Highlights")
  cc = st.columns(3)
  
  with cc[0]:
    # 'good', 'bad', 'neutral' sentiment to auto color the card
    hc.info_card(title='Loyal Customers', content=not_churns, sentiment='good')
    hc.info_card(title='Avg Monthly Charges', content = df.MonthlyCharges.mean(), sentiment='good')

  with cc[1]:
    hc.info_card(title='Churn Rate', content=churn_rate, sentiment='bad')
    hc.info_card(title='Total Charges', content = df.TotalCharges.sum(), sentiment='good')

  with cc[2]:
    hc.info_card(title='Churn Customers',content=churns, sentiment='bad')
    hc.info_card(title='Unique Customers', content=len(df.customerID.unique()), sentiment='neutral')
    
    
  
  # Charts sections 
  chart = functools.partial(st.plotly_chart, use_container_width=True)

  # Bar charts for gender and payment methods 
  cc = st.columns(2)
  
  with cc[0]:
     st.subheader("Churns by gender")
     fig = px.histogram(
          df,
          y="Churn",
          x="gender",
          color="gender"
          
          #color_discrete_sequence=px.colors.sequential.Greens,
          
        )
     fig.update_layout(barmode="group", xaxis={"categoryorder": "total descending"})
     chart(fig)

  with cc[1]:
     st.subheader("Churns by Contract")
     fig = px.histogram(
          df,
          y="Churn",
          x="PaymentMethod",
          color="PaymentMethod"
          
          #color_discrete_sequence=px.colors.sequential.Greens,
          
        )
     fig.update_layout(barmode="group", xaxis={"categoryorder": "total descending"})
     chart(fig)


  # Scatter plot for 3 numeric features with Churn label 

  st.subheader("How Churn decision is affected by each of Tenure, Monthly Charges, and Total Charges?")
  fig = px.scatter_matrix(df, dimensions=['Tenure','MonthlyCharges','TotalCharges'], color="Churn")

  chart(fig)


  # Pie charts section 
  cc= st.columns(2)
  # Contract Type pie chart
  with cc[0]:

     st.subheader("Churns by Contract Type")
     fig = px.pie(df.loc[df["Churn"]==1], values="Churn", names="Contract")
     fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
     chart(fig)
 
  # internet Service pie chart
  with cc[1]:

     st.subheader("Churns by Internet Service")
     fig = px.pie(df.loc[df["Churn"]==1], values="Churn", names="InternetService")
     fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
     chart(fig)

  



# Input section will take users input for a specific customer. 
# This direct input will be fir in the ML logistic regression model loaded to predict if this customer will churn or no. 
# The output is showna s an info card containing the predicted label for the tested customer. 
     
@app.addapp()
def Input():
  st.session_state = 0
  company_data = open("/Users/bothainaa/Desktop/Streamlit Churn App/Train_telco.csv", "r")
  df = pd.read_csv(company_data)
  df = manipulate_data(df)
  st.info("Input data for a customer to predict if he/she is most probably to churn or not")
  cc = st.columns(2)

  with cc[0]:
  
   gender = st.radio("Gender", list(df.gender.unique()))
   SeniorCitizen = st.radio("Senior Citizen", list(df.SeniorCitizen.unique()))
   Partner = st.radio("Partner", list(df.Partner.unique()))
   MultipleLines = st.radio("Multiple Lines", list(df.MultipleLines.unique()))
   PhoneService = st.radio("Phone Service", list(df.PhoneService.unique()))
   


  with cc[1]:

   PaymentMethod = st.selectbox("Payment method", list(df.PaymentMethod.unique()))
   InternetService = st.selectbox("Internet Services", list(df.InternetService.unique()))
   Contract = st.selectbox("Contract", list(df.Contract.unique()))
   Tenure = st.text_input("Tenure", "0")
   MonthlyCharges = st.slider("Monthly Charges",min_value=0, max_value=10000)
   TotalCharges = st.slider("Total Charges",min_value=0, max_value=100000)

  
  predict_btn = st.button("Predict")

  if predict_btn:

      label = input_churn_predictions(gender, SeniorCitizen, Partner, Tenure, PhoneService, MultipleLines, InternetService, Contract,
                   PaymentMethod, MonthlyCharges, TotalCharges )

      if label:
        hc.info_card(title="You are most Likely to Churn", content= int(label), sentiment='bad')
      else:
        hc.info_card(title="You are a Loyal Customer", content= int(label), sentiment='good')




  

# The batch preiction section is a function that calls the ML logistic regression model, 
# fit the input data which is in this section batch of customers, and then output a table having the Churn label.
#  You can download the table into a csv file. 

@app.addapp()
def Batch_Prediction():
    st.session_state = 0
    st.subheader("Upload your CSV from Telco")
    uploaded_data = st.file_uploader(
        "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=False
    )

    if uploaded_data is None:
        st.info("Using example data. Upload a file above to use your own data!")
        uploaded_data = open("/Users/bothainaa/Desktop/Streamlit Churn App/Train_telco.csv", "r")
    else:
        st.success("Uploaded your file!")
  

    # Raw data expander 
    df = pd.read_csv(uploaded_data)
    with st.expander("Raw Dataframe"):
        st.write(df)

    # Predicted data expander 
    df = churn_predictions_batch(df)
    with st.expander("Predicted Churn Data"):
        df = manipulate_data(df)
        st.write(df)



    col1, col2, col3 = st.columns(3)
    # Download button 
    with col2:
        
        st.download_button(
          label="Download data as CSV",
          data=convert_df(df),
          file_name='Predicted_Churn_Customer_Telco.csv',
          mime='text/csv',
     
         )

    

  



#This is the Churned Customers profiling section that will output all customers that already churned in table.
#  The table includes their personal informaion for the CRM purposes as of targeting them with email campaigns, specific messages offering them some promotions to retain them. 
# The table can be filtered by a specific customer and downlaoded as a batch.

@app.addapp()
def Churn_Customers():
  st.session_state = 0
  st.info("This is the Churned Customers Profiling section")
  st.subheader("Inspect Churned Customer Information!")
  st.markdown('<i class="fa face-meh"></i>', unsafe_allow_html=True)
  company_data = open("/Users/bothainaa/Desktop/Streamlit Churn App/Train_telco.csv", "r")
  df = pd.read_csv(company_data)
  df_churned = df.loc[df["Churn"] == 1]
  df_churned = manipulate_data(df_churned)
  customers_all = ["All"]
  customer_ids = list(df_churned.customerID.unique())
  customers = customers_all + customer_ids
    
  customer_to_inspect = st.selectbox("Select a Customer ", customers)
  if customer_to_inspect == "All":
        customer_data = df_churned[["customerID","gender","MonthlyCharges", "TotalCharges", "PaymentMethod", "Churn"]]
  else:
        customer_data = df_churned[["customerID","gender","MonthlyCharges", "TotalCharges", "PaymentMethod", "Churn"]].loc[df["customerID"] == customer_to_inspect]
  

  # Download button 
  st.download_button(
          label="Download data as CSV",
          data=convert_df(customer_data),
          file_name='Predicted_Churn_Customer_Telco.csv',
          mime='text/csv',
     
         )
  st.table(customer_data.style.applymap(highlight_churn, subset = ["Churn"]))
  


#This is the Loyal customers profiling function that will output a table including all base data customers that didn't churn.
# you have the option to filter for a specific customer and download a batch of customers for CRM purposes. 
# The function will take only the personal information fields from teh dataset and output them in table highlighting teh column Churn 
@app.addapp()
def Not_Churn_Customers():
  st.session_state = 0
  st.info("This is the Loyal Customers Profiling section")
  st.subheader("Inspect Loyal Customer Information!")
  st.markdown('<div><i class="fas fa-smile"></i></div>', unsafe_allow_html=True)
  company_data = open("/Users/bothainaa/Desktop/Streamlit Churn App/Train_telco.csv", "r")
  df = pd.read_csv(company_data)
  df_not_churned = df.loc[df["Churn"] == 0]
  df_not_churned = manipulate_data(df_not_churned)
  customers_all = ["All"]
  customer_ids = list(df_not_churned.customerID.unique())
  customers = customers_all + customer_ids
    
  customer_to_inspect = st.selectbox("Select a Customer ", customers)
  if customer_to_inspect == "All":
        customer_data = df_not_churned[["customerID","gender","MonthlyCharges", "TotalCharges", "PaymentMethod", "Churn"]]
  else:
        customer_data = df_not_churned[["customerID","gender","MonthlyCharges", "TotalCharges", "PaymentMethod", "Churn"]].loc[df["customerID"] == customer_to_inspect]
  
  # Download button 
  st.download_button(
          label="Download data as CSV",
          data=convert_df(customer_data),
          file_name='Predicted_Churn_Customer_Telco.csv',
          mime='text/csv',
     
         )
  st.table(customer_data.style.applymap(highlight_churn, subset = ["Churn"]))
  


# Adding each menu item app to the main HydraApp then running it. 
if __name__ == '__main__':
  
  if menu_id == "Analysis Dashboard":
     app.add_app("Home", app= Home())
     app._user_session_params
     
  else:
    if menu_id == "Input Data for Churn Prediction":
      app.add_app("Input Data For prediction", app= Input())
      app._user_session_params

    else:
      if menu_id == "Batch Prediction":
        app.add_app("Batch Prediction", app=Batch_Prediction())
        app._user_session_params
        
      else:
          if menu_id == "Churn":
            app.add_app("Churned Customers Profiling", app=Churn_Customers())
            app._user_session_params
            
          else: 
            if menu_id == "Not Churn":
              app.add_app("Loyal Customers Profiling", app=Not_Churn_Customers())
              app._user_session_params
              

  app.run()



