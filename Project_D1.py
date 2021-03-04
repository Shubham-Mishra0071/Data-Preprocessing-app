import streamlit as st
import pandas as pd
import numpy as np
import xlrd
import xlsxwriter
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import base64
from io import BytesIO
import cx_Oracle
import re
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from sqlalchemy import types, create_engine
import urllib.request




st.markdown("<h1 style='text-align: center; color: Light gray;'>Data Preprocessing App</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Data import</h3>", unsafe_allow_html=True)



st.markdown(
"""
<style>
.reportview-container {
    background: url("https://media.giphy.com/media/l378c04F2fjeZ7vH2/giphy.gif")
    
    
    
    
    
    
}
.sidebar .sidebar-content {
    background: url("https://media.giphy.com/media/l378c04F2fjeZ7vH2/giphy.gif")
}
</style>
""",
unsafe_allow_html=True
)

# Headings






temp='\\temp.csv'

path=os.getcwd()
path=path+temp












    
    
    
    
    
    
    
    # All Functions


        
        
        
        
    
def get_table_download_link_csv(df):
    try:
        
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframes
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(
            csv.encode()
        ).decode()  # some strings <-> bytes conversions necessary here
        return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df



def to_excel(df):
    try:
        
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df



def get_table_download_link_xlsx(df):
    try:
        
        
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        val = to_excel(df)
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="dataprep.xlsx">Download xlsx file</a>' # decode b'abc' => abc
    
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df


    

def mvt_median(df):
    try:
    
        clean_df=(df.fillna(df.median()))
        clean_df.fillna(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(df.dtypes)
        st.info("The Percenatge of Value Missing in Given Data is : {:.2f}%".format(((df.isna().sum().sum())/(df.count().sum())*100)))
        st.info("Data to be treated using MEDIAN : {}".format(list(dict(df.median()).keys())))
        st.info('Shape of dataframe (Rows, Columns):{} '.format(df.shape))
        st.write('Data description : ',df.describe())
        st.info("Only Numerical missing values will be treated using Median ")
        st.info("categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.line_chart(clean_df)
        return clean_df
    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
    
    

    
    

def mvt_mode(df):
    try:
        cat_col=list(df.select_dtypes(include ='object').columns)
        st.info("The Percentage of Value Missing in Given Data is : {:.3f}%".format((df[cat_col].isna().sum().sum())/(df.count().sum())*100))
        st.info("\nThe Percenatge of Value Missing in Given Data is :\n{}".format((df[cat_col].isnull().sum()*100)/df.shape[0]))
        clean_df=(df.fillna(df.select_dtypes(include ='object').mode().iloc[0]))
        st.dataframe(clean_df)
        st.info("\nData to be treated using MODE : {}".format(cat_col))
        st.write('Shape of dataframe (Rows, Columns): ',df.shape)
        st.write('Data description :\n',df.describe(include ='object'))
        st.info("Only categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.info("You can head to Mean or Median to treat the Numerical Missing Value")
        st.line_chart(clean_df)
        return clean_df
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        st.write("This can happen if there is no categorical data to treat")
        return df




def ot_iqr(df,column_name):
    
    try:
        
        
    
        #column_name="Marks_Grad"

        if column_name:



            q1 = df[column_name].quantile(0.25)
            q3 = df[column_name].quantile(0.75)
            IQR = q3 - q1
            lower_limit = q1 - 1.5*IQR
            upper_limit = q3 + 1.5*IQR
            removed_outlier = df[(df[column_name] > lower_limit) & (df[column_name] < upper_limit)]   
            st.dataframe(removed_outlier)
            st.write("Percentile Of Dataset :\n ", df.describe())
            st.info('Size of dataset after outlier removal')
            st.write(removed_outlier.shape)
            st.line_chart(removed_outlier)
            return removed_outlier
        
        
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

    
    
    
def z_score(df,column_name):
    
    try:
        
    
        if column_name:

            df['z-score'] = (df[column_name]-df[column_name].mean())/df[column_name].std() #calculating Z-score
            outliers = df[(df['z-score']<-1) | (df['z-score']>1)]   #outliers
            removed_outliers = pd.concat([df, outliers]).drop_duplicates(keep=False)   #dataframe after removal 
            st.dataframe(removed_outliers)
            st.write("Percentile Of Dataset :\n ", df.describe())
            st.write('Number of outliers : {}'.format(outliers.shape[0])) #number of outliers in Given Dataset
            st.info('Size of dataset after outlier removal')
            st.write(removed_outliers.shape)
            st.line_chart(removed_outliers)
            return removed_outliers
        
        
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df


    
    
        

    
    
    
    

    
    
    
    
    
    
    
    
    




    
    
    












    




        

    




    
    
        
        
        

    
    
    

    
                
                
            
                    
# MVT Options 


def mvt_options(df):
    
    
    
    
    selection=st.sidebar.radio('Choose method',('Traditional method of Mean/Median and Mode','K-Nearest Neighbors imputation'))
    if selection=='Traditional method of Mean/Median and Mode':
        select=st.sidebar.radio('choose method',('Mean','Median','Mode'))
        if select=='Mean':
            try:
                
                    
                df = pd.read_csv(path)
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(missing_values=np.nan,strategy='mean')
                    
                            
                            
                COLUMN = st.multiselect('Please select the columns',df.columns)
                
                df[COLUMN] =imp.fit(df[COLUMN]).transform(df[COLUMN])
                df.to_csv(path,index=False)
                st.dataframe(df)
                return df
            except ValueError:
                    
                st.error('at least one array or dtype is required')
                return df        
        elif select=='Median':
            try:
                
                df = pd.read_csv(path)
                from sklearn.impute import SimpleImputer
                            
                                
                COLUMN=st.multiselect('Please select the columns',df.columns)
                            
                df[COLUMN] =imp.fit(df[COLUMN]).transform(df[COLUMN])
                df.to_csv(path,index=False)  
                st.dataframe(df) 
                return df
            except ValueError:
                    
                st.error('at least one array or dtype is required')
                            
        elif select=='Mode':
            try:
                
                df = pd.read_csv(path)
                from sklearn.impute import SimpleImputer
                            
                                    
                COLUMN=st.multiselect('Please select the columns',df.columns)
                
                df[COLUMN] =imp.fit(df[COLUMN]).transform(df[COLUMN])
                df.to_csv(path,index=False)
                st.dataframe(df)
                return df
            except ValueError:
                st.error('at least one array or dtype is required')
                            
    elif selection=='K-Nearest Neighbors imputation':
        
            
        try:
            
            df = pd.read_csv(path)    
            from sklearn.impute import KNNImputer
            n=st.slider('Please enter the value on n',2,10)
                        
            imputer = KNNImputer(n_neighbors=n)
                       
                                    
            COLUMN=st.multiselect('Please select the columns',df.columns)
            
            df[COLUMN] = imputer.fit_transform(df[COLUMN])
            df.to_csv(path,index=False)
            st.dataframe(df)
            return df
        except ValueError:
            st.error('at least one array or dtype is required')
            return df
            
            

# Outliers Function

def outlier_function():
    try:
        
        option_o=("IQR","Z-Score")
        o_f_selection = st.sidebar.radio("Choose a Outlier Treatment Method",option_o)
        if o_f_selection == "IQR":
            st.sidebar.write('you selected IQR')
            if st.sidebar.button('Process IQR'):
                df = pd.read_csv(path)
                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3-Q1
                df[~((df < (Q1-1.5*IQR)) | (df >(Q3+1.5*IQR))).any(axis=1)]
                return df
                

        elif o_f_selection == "Z-Score":
            st.sidebar.write('you selected Z-Score')
            if st.sidebar.button('Process Z-Score'):
                column_name=st.text_input("Enter the name of Column fom which outlier will be removed")
                st.info("You can find the list of columns below")
                df = pd.read_csv(path)
                st.write(df.columns)
                if st.sidebar.button("Process Z-Score"):
                    df = pd.read_csv(path)
                    if column_name in df.columns:
    
                        df=z_score(df,column_name)
                        df.to_csv(path, index=False)
                        return df
                    else:
                        st.info("This Column Name is Not Present")

                    
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

        
        
    
    

    
# feature Scaling options



def fso(df):
    
    
    F=st.sidebar.radio('Options',('Standard_Scalar','Min_Max_Scalar','Robust_Scalar','Max_Absolute_scalar'))
    
    if F=='Standard_Scalar':
        try:
            
            df = pd.read_csv(path)
            from sklearn.preprocessing import StandardScaler
                    
                    
            COLUMN=st.multiselect('Please select the columns',df.columns)
            sc_x = StandardScaler()
            df[COLUMN] = sc_x.fit_transform(df[COLUMN])
            df.to_csv(path,index=False)
            st.write(df)
            return df
        except ValueError:
            st.info('at least one array or dtype is required')
    elif F=='Min_Max_Scalar':
        try:
            from sklearn.preprocessing import MinMaxScaler
            df = pd.read_csv(path)        
            COLUMN=st.multiselect('Please select the columns',df.columns)
            x_mm = MinMaxScaler()
            df[COLUMN] = x_mm.fit_transform(df[COLUMN])
            df.to_csv(path,index=False)
            st.write(df)
            return df
        except ValueError:
            st.info('at least one array or dtype is required')
    elif F=='Robust_Scalar':
        try:
            from sklearn.preprocessing import RobustScaler
            df = pd.read_csv(path)
            COLUMN=st.multiselect('Please select the columns',df.columns)
            rob_x = RobustScaler()
            df[COLUMN] = rob_x.fit_transform(df[COLUMN])
            df.to_csv(path,index=False)
            st.write(df)
            return df
        except ValueError:
            st.info('at least one array or dtype is required')
    else:
        try:
            from sklearn.preprocessing import MaxAbsScaler
            df = pd.read_csv(path)        
            COLUMN=st.multiselect('Please select the columns',df.columns)
            max_x = MaxAbsScaler()
            df[COLUMN] = max_x.fit_transform(df[COLUMN])
            df.to_csv(path,index=False)
            st.write(df)
            return df
        except ValueError:
            st.info('at least one array or dtype is required')
            

    
    

            
    
    
    
    
    
def upload_xlsx(uploaded_file):
    
    try:
        
    
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            return df
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

def upload_csv(uploaded_file):
    
    try:
        
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            df.to_csv(path, index=False)
            return df

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    
    
    
def mail():
    
    try:
        
        
        mail_content = '''Hello,
        This is a Data Pre Processed File.
        Please see the attachmet below .
        Thank You for using our app
        '''

        #os.chdir(path)
        #The mail addresses and password
        file_name='pass.txt'
        if os.path.exists(file_name):
            with open('pass.txt', 'r') as file:  
                sender_pass=file.read()
                file.close()

        else:
            urllib.request.urlretrieve("https://drive.google.com/u/0/uc?id=1tan_wJsUqOtBTJv1lrwpqqJYgdVJY1td&export=download", "pass.txt")
            with open('pass.txt', 'r') as file: 
                sender_pass=file.read()
                file.close()

        sender_address = 'dpreprocessing@gmail.com'
  
        regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
        receiver_address = st.text_input("Please Enter The Email Address")
        if receiver_address:
            if(re.search(regex,receiver_address)):
                #Setup the MIME
                message = MIMEMultipart()
                message['From'] = sender_address
                message['To'] = receiver_address
                message['Subject'] = 'Please see your processed file in attachment'
                #The subject line
                #The body and the attachments for the mail
                message.attach(MIMEText(mail_content, 'plain'))
                attach_file_name = path
                attach_file = open(attach_file_name) # Open the file as binary mode
                payload = MIMEBase('application', 'octate-stream')
                payload.set_payload((attach_file).read())
                encoders.encode_base64(payload) #encode the attachment
                #add payload header with filename
                payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
                message.attach(payload)
                #Create SMTP session for sending the mail
                session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
                session.starttls() #enable security
                session.login(sender_address, sender_pass) #login with mail_id and password
                text = message.as_string()
                session.sendmail(sender_address, receiver_address, text)
                session.quit()
                st.write('Mail Sent Successfully to {}'.format(receiver_address))

            else:
                st.warning("Please Enter a Valid Email Address")



    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        
    
            
            
            
            
            
# File Upload
def file_upload():
    
    try:
        
    
        f_option=('.Xlsx','.Csv','Oracle')
        f_select=st.sidebar.radio('Choose a file type',f_option)

        if f_select == '.Xlsx':

            uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")

            if uploaded_file:

                if st.sidebar.button('Upload File'):
                    df=upload_xlsx(uploaded_file)
                    return df



        elif f_select == '.Csv':
            uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

            if uploaded_file:
                if st.sidebar.button('Upload File'):
                    df=upload_csv(uploaded_file)
                    return df

        elif f_select == 'Oracle':

            st.info("Enter Oracle Database information")

            user=st.text_input("Enter User name ")
            passwd=st.text_input("Enter Password ", type="password")
            host=st.text_input("Enter Host Address")
            port=st.text_input("Enter Port number")
            query =st.text_input("Enter the query for the desired data")


            if st.button("Connect"):


                con_query="{}/{}@{}:{}/ORCL".format(user,passwd,host,port)

                con=cx_Oracle.connect(con_query)

                if con!=None:
                    st.info("Connection Established Successfully")
                    df = pd.read_sql(query,con)
                    st.dataframe(df)
                    df.to_csv(path, index=False)
                    return df


                    


        
    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
        
    
        

# Data export

def data_export(df):
    
    try:
        
    
    
        
        st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Data Export</h3>", unsafe_allow_html=True)
        fd_option=('.Xlsx','.Csv','Oracle','Email')
        fd_select=st.sidebar.radio('Choose a file type to download',fd_option)

        if fd_select == '.Csv':
            if st.sidebar.button('Download Csv'):
                df = pd.read_csv(path)

                st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
                return 0


        elif fd_select == '.Xlsx':
            if st.sidebar.button('Download Xlsx'):
                df = pd.read_csv(path)
                st.sidebar.markdown(get_table_download_link_xlsx(df), unsafe_allow_html=True)
                return 0


        elif fd_select == 'Oracle':
            st.info("Enter Oracle Database information")

            users=st.text_input("Enter Users name ")
            passwd=st.text_input("Enter Password ", type="password")
            host=st.text_input("Enter Host Address")
            port=st.text_input("Enter Port number")
            table=st.text_input("Enter the name of table to create, if table exist it'll be replaced")
            if st.button("Connect"):
                df = pd.read_csv(path)
                conn = create_engine('oracle+cx_oracle://{}:{}@{}:{}/ORCL'.format(users,passwd,host,port))
                df.to_sql('{}'.format(table), conn, if_exists='replace')
                
                if conn!=None:
                    st.info("Connection Established Successfully and Table Inserted")



        elif fd_select == "Email":
            mail()


    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df

# Give main options


def main_option():
    
    try:
        
    
        option=('Missing Value Treatment', 'Outlier Treatment', 'Feature Scaling')

        option_select = st.sidebar.radio('What would you like to do?',option)

        return option_select

    
    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        
    
                        

def main():
    
    try:
        
    
        
        df=file_upload()

        m_option = main_option()

        if m_option == 'Missing Value Treatment':

            df=mvt_options(df)


        elif m_option == 'Outlier Treatment':

            outlier_function()

        elif m_option == 'Feature Scaling':

            fso(df)


        data_export(df)

    except Exception as e:
        st.write("Oops!", e.__class__, "occurred.")
        return df
    

main()