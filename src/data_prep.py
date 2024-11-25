

import pandas as pd
import numpy as np


df_defect_rate = pd.read_csv("../Masked_Renamed_Factory_Sample/Defect Rates.csv")
df_step1_terminal = pd.read_csv("../Masked_Renamed_Factory_Sample/Step1_Mount_Terminals.csv")
df_step1_resin = pd.read_csv("../Masked_Renamed_Factory_Sample/Step1_Mount_Terminal_Resin.csv")
df_step2_wind_wire = pd.read_csv("../Masked_Renamed_Factory_Sample/Step2_Wind_Wire.csv")
df_step3_peel_wire = pd.read_csv("../Masked_Renamed_Factory_Sample/Step3_Peel_Wire.csv")
df_step4_check_alignment = pd.read_csv("../Masked_Renamed_Factory_Sample/Step4_Check_Alignment.csv")


print(df_defect_rate.shape)
print(df_step1_terminal.shape)
print(df_step1_resin.shape)
print(df_step2_wind_wire.shape)
print(df_step3_peel_wire.shape)
print(df_step4_check_alignment.shape)


print(df_defect_rate.isnull().sum())
print(" ")
print(df_step1_terminal.isnull().sum())

print(" ")
print(df_step1_resin.isnull().sum())
print(" ")

print(df_step2_wind_wire.isnull().sum())
print(" ")
print(df_step3_peel_wire.isnull().sum())
print(" ")
print(df_step4_check_alignment.isnull().sum())


# - We observer that there no missing value in the dataset



print(df_defect_rate.nunique())
print(" ")
print(df_step1_terminal.nunique())

print(" ")
print(df_step1_resin.nunique())
print(" ")

print(df_step2_wind_wire.nunique())
print(" ")
print(df_step3_peel_wire.nunique())
print(" ")
print(df_step4_check_alignment.nunique())


# - Observe the measurement column has unique in dataset




df_step1_resin[["DateTime",'Time',"MeasurementCount"]].sort_values("MeasurementCount",ascending=True)[20:40]


# - we can observe that MeasurementCount increase with time
# - data is at model second level for each minutes 
# - time column dosen't have Hour

# In[7]:


df=df_defect_rate[::-1].reset_index(drop=True)
# Function to parse time into minutes and seconds
def parse_time(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes), float(seconds)

# Parse Time column into minutes and seconds
df[['Minutes', 'Seconds']] = df['Time'].apply(lambda x: pd.Series(parse_time(x)))

# Reverse the dataset to make transition checks easier
data = df[::-1].reset_index(drop=True)

# Initialize transition flag
flags = []

# Iterate through rows to identify transitions
for i in range(len(df)):
    if i > 0:
        prev_minutes, prev_seconds = df.loc[i - 1, ['Minutes', 'Seconds']]
        curr_minutes, curr_seconds = df.loc[i, ['Minutes', 'Seconds']]
        
        # # Check for transition from 59:58.x to 00:00.x
        # if prev_minutes == 0 and prev_seconds <= 1 and curr_minutes == 59 and curr_seconds >= 58:
        if prev_minutes == 59 and prev_seconds >= 58 and curr_minutes == 0 and curr_seconds <= 1:    
            flags.append(1)  # Mark transition with a flag
        else:
            flags.append(0)
    else:
        flags.append(0)  # First row, no transition

# Add Flag column to the DataFrame
df['Transition_Flag'] = flags


indx_lst = df[df['Transition_Flag']==1].index.to_list()
# Add the starting and ending indices for the full range
start_indices = [0] + [i + 1 for i in indx_lst]
end_indices = indx_lst + [None]  # None to indicate the end of the range

# Generate index ranges
index_ranges = [(start, end) for start, end in zip(start_indices, end_indices)]

# Print the ranges
cnt=0
for start, end in index_ranges:
    print(f"Range: [{start}:{end}]")
    # print('0'+str(cnt)+":"+df['Time'])
    df.loc[start:end,'Time']='0'+str(cnt)+":"+df['Time']
    cnt+=1

df['DateTime']=pd.to_datetime(df['Date']+" "+df['Time'])#,format='%m/%d/%Y %H:%M:%S.%f')
df['month'] =df['DateTime'].dt.month
df['year'] =df['DateTime'].dt.year
df['hour']=df['DateTime'].dt.hour

# selecting the required column for moving for problem solving
df = df[['DateTime', 'month', 'year', 'hour', 'Minutes', 'Seconds','Defect Rate']]
#Removing the % sign for futher calculation
df['Defect_Rate']=df['Defect Rate'].apply(lambda x:x.split("%")[0]).astype(int)/100

df=df.drop(columns=['Defect Rate'],axis=1)


# In[8]:


df.head()


# In[9]:


def drop_columns(df):
    temp1 = df.nunique().reset_index()
    temp1.columns = ['columns','cnt']
    drop_list = temp1.loc[temp1['cnt']==1,'columns'].to_list()  # selecting the columns with low variance
    drop_list.append("Time")
    df = df.drop(columns=drop_list,axis=1)
    df.sort_values("MeasurementCount",ascending=True,inplace=True)
    return df


# In[10]:


df_step1_terminal = drop_columns(df_step1_terminal)
df_step1_resin = drop_columns(df_step1_resin)
df_step2_wind_wire = drop_columns(df_step2_wind_wire)
df_step3_peel_wire = drop_columns(df_step3_peel_wire)
temp_date_time = df_step4_check_alignment[['DateTime',"Time"]]
df_step4_check_alignment = drop_columns(df_step4_check_alignment)

df_step4_check_alignment = pd.concat([temp_date_time,df_step4_check_alignment],axis=1)


# In[34]:


combined_df = pd.concat([df_step1_terminal,df_step1_resin,df_step2_wind_wire,df_step3_peel_wire,df_step4_check_alignment],axis=1)


# In[44]:


preprocess_data =combined_df.reset_index(drop=True)


# In[46]:


# Function to parse time into minutes and seconds
def parse_time(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes), float(seconds)

# Parse Time column into minutes and seconds
preprocess_data[['Minutes', 'Seconds']] = preprocess_data['Time'].apply(lambda x: pd.Series(parse_time(x)))

# Initialize transition flag
flags = []

# Iterate through rows to identify transitions
for i in range(len(data)):
    if i > 0:
        prev_minutes, prev_seconds = preprocess_data.loc[i - 1, ['Minutes', 'Seconds']]
        curr_minutes, curr_seconds = preprocess_data.loc[i, ['Minutes', 'Seconds']]
        
        # Check for transition from 59:58.x to 00:00.x
        if prev_minutes == 59 and prev_seconds >= 58 and curr_minutes == 0 and curr_seconds <= 1:
            flags.append(1)  # Mark transition with a flag
        else:
            flags.append(0)
    else:
        flags.append(0)  # First row, no transition

# Add Transition_Flag column to the DataFrame
preprocess_data['Transition_Flag'] = flags

# Drop unnecessary columns if needed
preprocess_data = preprocess_data.drop(columns=['Minutes', 'Seconds'])


# In[47]:


indx_lst_pre = preprocess_data[preprocess_data['Transition_Flag']==1].index.to_list()
indx_lst_pre
# Add the starting and ending indices for the full range
start_indices = [0] + [i + 1 for i in indx_lst_pre]
end_indices = indx_lst_pre + [None]  # None to indicate the end of the range

# Generate index ranges
index_ranges = [(start, end) for start, end in zip(start_indices, end_indices)]

# Print the ranges
cnt=0
for start, end in index_ranges:
    print(f"Range: [{start}:{end}]")
    # print('0'+str(cnt)+":"+df['Time'])
    preprocess_data.loc[start:end,'Time']='0'+str(cnt)+":"+preprocess_data['Time']
    cnt+=1


# In[ ]:


preprocess_data['DateTime']=pd.to_datetime(preprocess_data['DateTime']+" "+preprocess_data['Time'])#,format='%m/%d/%Y %H:%M:%S.%f')
preprocess_data['month'] =preprocess_data['DateTime'].dt.month
preprocess_data['year'] =preprocess_data['DateTime'].dt.year
preprocess_data['hour']=preprocess_data['DateTime'].dt.hour
preprocess_data = preprocess_data.drop(columns=['Time'],axis=1)


# In[50]:


preprocess_data.head()


# In[52]:


preprocess_data = preprocess_data.drop(columns=['Time'],axis=1)


# In[53]:


import os


#os.makedirs("../preprocessed_data/") # uncomment the code to create folder



df.to_csv("../preprocessed_data/defect_rate_preprocessed.csv",index=False)
preprocess_data.to_csv("../preprocessed_data/combined_processed_data.csv",index=False)





