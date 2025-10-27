"""
HCI UX Dataset Generator
Generates realistic, multi-method UX datasets for teaching UX research, HCI, and cognitive psychology.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
RANDOM_SEEDS = {
    'small': 42,
    'medium': 123,
    'large': 456
}

def inject_missing_values(df, missing_rate=0.02):
    """Inject random missing values into dataframe"""
    df = df.copy()
    n_missing = int(len(df) * len(df.columns) * missing_rate)
    for _ in range(n_missing):
        row_idx = np.random.randint(0, len(df))
        col_idx = np.random.randint(0, len(df.columns))
        df.iloc[row_idx, col_idx] = np.nan
    return df

def inject_outliers(values, outlier_rate=0.02):
    """Inject outliers into a numeric array"""
    values = values.copy()
    n_outliers = int(len(values) * outlier_rate)
    outlier_indices = np.random.choice(len(values), size=n_outliers, replace=False)
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            values[idx] = values[idx] * (2 + np.random.random() * 3)  # Large positive outlier
        else:
            values[idx] = values[idx] * (0.1 - np.random.random() * 0.05)  # Small outlier
    return values

def add_noise(values, noise_level=0.05):
    """Add random noise to values"""
    noise = np.random.normal(0, noise_level * np.std(values), size=len(values))
    return values + noise

# Dataset 1: Survey & Questionnaire Data
def generate_survey_data(size, seed):
    np.random.seed(seed)
    n = size
    
    data = {
        'Participant_ID': [f'P{i+1:04d}' for i in range(n)],
        'Age': np.random.normal(35, 12, n).astype(int).clip(18, 75),
        'Gender': np.random.choice(['M', 'F', 'Other', 'Prefer not to say'], n, p=[0.45, 0.48, 0.04, 0.03]),
        'Device_Type': np.random.choice(['iOS', 'Android', 'Desktop'], n, p=[0.5, 0.35, 0.15]),
        'Experience_Level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n, p=[0.3, 0.5, 0.2]),
    }
    
    # Task performance
    data['Task_Success'] = np.random.choice([0, 1], n, p=[0.15, 0.85])
    data['Completion_Time_s'] = np.random.lognormal(4.2, 0.8, n)
    
    # SUS Questions (1-5 scale)
    for i in range(1, 11):
        data[f'SUS_Q{i}'] = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3])
    
    # UEQ dimensions (1-7 scale)
    data['UEQ_Attractiveness'] = np.random.choice([3, 4, 5, 6, 7], n, p=[0.05, 0.15, 0.3, 0.35, 0.15])
    data['UEQ_Efficiency'] = np.random.choice([3, 4, 5, 6, 7], n, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    data['UEQ_Perspicuity'] = np.random.choice([3, 4, 5, 6, 7], n, p=[0.05, 0.15, 0.35, 0.35, 0.1])
    
    # NASA-TLX dimensions (0-21)
    data['NASA_TLX_Mental'] = np.random.lognormal(2.5, 0.6, n).clip(0, 21)
    data['NASA_TLX_Temporal'] = np.random.lognormal(2.3, 0.5, n).clip(0, 21)
    data['NASA_TLX_Frustration'] = np.random.exponential(5, n).clip(0, 21)
    
    # Trust and related metrics (1-7)
    data['Trust_Score'] = np.random.beta(6, 2, n) * 6 + 1
    data['Ease_of_Use'] = np.random.beta(7, 2, n) * 6 + 1
    data['Willingness_to_Reuse'] = np.random.beta(5, 3, n) * 6 + 1
    
    # Correlate variables
    data['Trust_Score'] += (data['Task_Success'] - 0.5) * 2
    data['Ease_of_Use'] += (data['Task_Success'] - 0.5) * 1.5
    data['NASA_TLX_Frustration'] -= data['Task_Success'] * 5
    
    # Attention check
    data['Attention_Check_Passed'] = np.random.choice([0, 1], n, p=[0.12, 0.88])
    
    df = pd.DataFrame(data)
    
    # Inject missing values
    df = inject_missing_values(df, 0.02)
    
    # Add outlier to completion time
    df['Completion_Time_s'] = inject_outliers(df['Completion_Time_s'].fillna(0), 0.02)
    
    return df

# Dataset 2: Usability Testing Data
def generate_usability_test_data(size, seed):
    np.random.seed(seed)
    n = size
    tasks = ['Checkout', 'Product Search', 'Account Setup', 'Checkout', 'Payment', 'Account Setup']
    
    data = {
        'Participant_ID': [],
        'Task_Name': [],
        'Task_Success': [],
        'Completion_Time_s': [],
        'Error_Count': [],
        'Error_Type': [],
        'Frustration_Level': [],
        'Help_Requested': [],
        'Moderator': [],
        'Device_Type': [],
        'Satisfaction_Score': [],
    }
    
    for i in range(n):
        p_id = f'P{i+1:04d}'
        task = np.random.choice(tasks)
        success = np.random.choice([0, 1], p=[0.2, 0.8])
        
        # Correlate completion time with success
        base_time = np.random.lognormal(4, 0.8)
        comp_time = base_time if success else base_time * 1.5
        
        # Error count (Poisson)
        error_count = np.random.poisson(0.5) if success else np.random.poisson(2.5)
        error_types = ['Validation', 'Navigation', 'Input', 'System']
        error_type = np.random.choice(error_types) if error_count > 0 else 'None'
        
        data['Participant_ID'].append(p_id)
        data['Task_Name'].append(task)
        data['Task_Success'].append(success)
        data['Completion_Time_s'].append(comp_time)
        data['Error_Count'].append(error_count)
        data['Error_Type'].append(error_type)
        data['Frustration_Level'].append(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.3, 0.2, 0.1]))
        data['Help_Requested'].append(np.random.choice([0, 1], p=[0.7, 0.3]))
        data['Moderator'].append(np.random.choice(['Mod_A', 'Mod_B', 'Mod_C']))
        data['Device_Type'].append(np.random.choice(['Desktop', 'Tablet', 'Mobile'], p=[0.5, 0.2, 0.3]))
        data['Satisfaction_Score'].append(np.random.choice([3, 4, 5, 6, 7], p=[0.05, 0.1, 0.25, 0.4, 0.2]))
    
    df = pd.DataFrame(data)
    
    # Correlate frustration with errors and success
    df['Frustration_Level'] = (df['Frustration_Level'] + 
                               df['Error_Count'] * 0.3 - 
                               df['Task_Success'] * 1.5).clip(1, 5).astype(int)
    
    df = inject_missing_values(df, 0.025)
    
    return df

# Dataset 3: Interaction / Telemetry Log Data
def generate_interaction_log_data(size, seed):
    np.random.seed(seed)
    n = size  # This will be rows, not users
    
    events = ['click', 'scroll', 'page_view', 'form_submit', 'click', 'hover']
    devices = ['Desktop', 'Mobile', 'Tablet']
    browsers = ['Chrome', 'Safari', 'Firefox', 'Edge']
    error_codes = ['None', '404', '500', 'Timeout', 'None', 'None']
    
    data = {
        'User_ID': [],
        'Session_ID': [],
        'Timestamp': [],
        'Event_Type': [],
        'Event_Target': [],
        'Click_X': [],
        'Click_Y': [],
        'Scroll_Depth_%': [],
        'Dwell_Time_ms': [],
        'Time_on_Page_s': [],
        'Session_Length_s': [],
        'Device_Type': [],
        'Browser': [],
        'Network_Speed_Mbps': [],
        'Error_Code': [],
        'Conversion': [],
    }
    
    n_users = n // 10  # ~10 events per user on average
    user_ids = [f'U{i+1:05d}' for i in range(n_users)]
    
    for i in range(n):
        user_id = np.random.choice(user_ids)
        session_id = f"{user_id}_S{np.random.randint(1, 4)}"
        
        # Generate timestamp
        base_time = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90))
        timestamp = base_time + timedelta(seconds=np.random.randint(0, 86400))
        
        event_type = np.random.choice(events)
        device = np.random.choice(devices)
        
        data['User_ID'].append(user_id)
        data['Session_ID'].append(session_id)
        data['Timestamp'].append(timestamp)
        data['Event_Type'].append(event_type)
        data['Event_Target'].append(f"element_{np.random.randint(1, 50)}")
        data['Click_X'].append(np.random.randint(0, 1920) if event_type in ['click', 'hover'] else np.nan)
        data['Click_Y'].append(np.random.randint(0, 1080) if event_type in ['click', 'hover'] else np.nan)
        data['Scroll_Depth_%'].append(np.random.randint(0, 101) if event_type == 'scroll' else np.nan)
        data['Dwell_Time_ms'].append(np.random.lognormal(7, 1.2) if event_type in ['hover', 'page_view'] else np.nan)
        data['Time_on_Page_s'].append(np.random.lognormal(3, 1))
        data['Session_Length_s'].append(np.random.lognormal(4, 0.8))
        data['Device_Type'].append(device)
        data['Browser'].append(np.random.choice(browsers))
        data['Network_Speed_Mbps'].append(np.random.lognormal(3.5, 0.8))
        data['Error_Code'].append(np.random.choice(error_codes))
        data['Conversion'].append(np.random.choice([0, 1], p=[0.95, 0.05]))
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.03)
    
    return df

# Dataset 4: Eye Tracking Data
def generate_eye_tracking_data(size, seed):
    np.random.seed(seed)
    n = size
    
    conditions = ['Design_A', 'Design_B', 'Control']
    aois = ['Header', 'Navigation', 'Content', 'CTA_Button', 'Footer', 'Sidebar']
    
    data = {
        'Participant_ID': [],
        'Stimulus_ID': [],
        'AOI_Label': [],
        'Fixation_Count': [],
        'Fixation_Duration_ms': [],
        'Dwell_Time_ms': [],
        'Time_to_First_Fixation_ms': [],
        'First_Fixation_Duration_ms': [],
        'Saccade_Count': [],
        'Gaze_Path_Length_px': [],
        'Average_Pupil_Diameter_mm': [],
        'Blink_Rate_per_s': [],
        'Entry_Time_s': [],
        'Exit_Time_s': [],
        'Condition': [],
    }
    
    n_participants = n // 5  # ~5 AOIs per participant
    
    for i in range(n):
        p_id = f'P{i%n_participants+1:04d}'
        condition = np.random.choice(conditions)
        aoi = np.random.choice(aois)
        stimulus = f"Stim_{np.random.randint(1, 21)}"
        
        fixation_count = int(np.random.lognormal(3, 0.8))
        dwell_time = np.random.lognormal(6.5, 1.2)
        ttff = np.random.lognormal(5, 1.5)
        pupil = np.random.normal(4.2, 0.6)
        
        data['Participant_ID'].append(p_id)
        data['Stimulus_ID'].append(stimulus)
        data['AOI_Label'].append(aoi)
        data['Fixation_Count'].append(fixation_count)
        data['Fixation_Duration_ms'].append(np.random.lognormal(5.5, 0.7))
        data['Dwell_Time_ms'].append(dwell_time)
        data['Time_to_First_Fixation_ms'].append(ttff)
        data['First_Fixation_Duration_ms'].append(np.random.lognormal(5, 0.6))
        data['Saccade_Count'].append(int(np.random.poisson(15)))
        data['Gaze_Path_Length_px'].append(np.random.lognormal(9, 1))
        data['Average_Pupil_Diameter_mm'].append(np.clip(pupil, 3, 6))
        data['Blink_Rate_per_s'].append(np.random.lognormal(-1, 0.5))
        data['Entry_Time_s'].append(np.random.uniform(0, 20))
        data['Exit_Time_s'].append(np.random.uniform(data['Entry_Time_s'][-1], 30))
        data['Condition'].append(condition)
    
    df = pd.DataFrame(data)
    
    # Correlate TTFF with task success (inverse)
    task_success = np.random.binomial(1, 0.8, len(df))
    df['Time_to_First_Fixation_ms'] = df['Time_to_First_Fixation_ms'] * (1 - task_success * 0.3)
    
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 5: EEG / GSR / Physiological Data
def generate_physiological_data(size, seed):
    np.random.seed(seed)
    n = size
    
    conditions = ['Game_A', 'Game_B', 'Rest']
    events = ['Action', 'Cutscene', 'Menu', 'Loading', 'Combat']
    
    data = {
        'Participant_ID': [],
        'Condition': [],
        'Task_ID': [],
        'EEG_Alpha': [],
        'EEG_Beta': [],
        'EEG_Theta': [],
        'EEG_Workload_Index': [],
        'EEG_Engagement_Index': [],
        'GSR_Peak_Count': [],
        'GSR_Peak_Amplitude': [],
        'Heart_Rate_BPM': [],
        'Valence': [],
        'Arousal': [],
        'Pupil_Diameter_mm': [],
        'Blink_Rate_per_min': [],
        'Timestamp': [],
        'Event_Label': [],
    }
    
    n_participants = n // 20  # ~20 measurements per participant
    
    base_time = datetime(2024, 1, 1)
    
    for i in range(n):
        p_id = f'P{i%n_participants+1:04d}'
        condition = np.random.choice(conditions)
        task = f"Task_{np.random.randint(1, 10)}"
        event = np.random.choice(events)
        
        timestamp = base_time + timedelta(seconds=i * 2)
        
        # Generate workload (influences other measures)
        workload = np.random.normal(50, 15)
        
        # EEG measures
        eeg_alpha = np.random.normal(30, 8) + workload * 0.2
        eeg_beta = np.random.normal(20, 6)
        eeg_theta = np.random.normal(25, 7) - workload * 0.1
        
        # Correlate measures
        hr = np.random.normal(70, 12) + workload * 0.3
        gsr_amplitude = np.random.lognormal(1.5, 0.8) + workload * 0.02
        engagement = np.random.normal(60, 15) + workload * 0.4
        
        data['Participant_ID'].append(p_id)
        data['Condition'].append(condition)
        data['Task_ID'].append(task)
        data['EEG_Alpha'].append(eeg_alpha)
        data['EEG_Beta'].append(eeg_beta)
        data['EEG_Theta'].append(eeg_theta)
        data['EEG_Workload_Index'].append(workload)
        data['EEG_Engagement_Index'].append(engagement)
        data['GSR_Peak_Count'].append(int(np.random.poisson(3)))
        data['GSR_Peak_Amplitude'].append(gsr_amplitude)
        data['Heart_Rate_BPM'].append(np.clip(hr, 50, 120))
        data['Valence'].append(np.random.normal(0, 1))
        data['Arousal'].append(np.random.normal(0, 1))
        data['Pupil_Diameter_mm'].append(np.random.normal(4.5, 0.7))
        data['Blink_Rate_per_min'].append(int(np.random.poisson(20)))
        data['Timestamp'].append(timestamp)
        data['Event_Label'].append(event)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.03)
    
    return df

# Dataset 6: Card Sorting / Tree Testing
def generate_card_sorting_data(size, seed):
    np.random.seed(seed)
    n = size
    
    cards = ['Home', 'About', 'Contact', 'Courses', 'Faculty', 'Research', 'Admissions', 'Campus']
    categories = ['Main_Nav', 'Academics', 'Administration', 'Resources']
    design_versions = ['Current', 'Proposed']
    
    data = {
        'Participant_ID': [],
        'Card_ID': [],
        'Card_Label': [],
        'Assigned_Category': [],
        'Category_Correct': [],
        'Confidence_Rating': [],
        'Task_Success': [],
        'Path_Length': [],
        'Time_on_Task_s': [],
        'Design_Version': [],
    }
    
    n_participants = n // 8  # 8 cards per participant
    
    for i in range(n):
        p_id = f'P{i%n_participants+1:04d}'
        card = np.random.choice(cards)
        category = np.random.choice(categories)
        version = np.random.choice(design_versions)
        
        # Determine correctness (design version affects it)
        correct = 1 if (version == 'Proposed') else np.random.choice([0, 1], p=[0.3, 0.7])
        
        confidence = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
        path_length = int(np.random.poisson(3))
        
        data['Participant_ID'].append(p_id)
        data['Card_ID'].append(f"Card_{i+1}")
        data['Card_Label'].append(card)
        data['Assigned_Category'].append(category)
        data['Category_Correct'].append(correct)
        data['Confidence_Rating'].append(confidence)
        data['Task_Success'].append(np.random.choice([0, 1], p=[0.15, 0.85]))
        data['Path_Length'].append(path_length)
        data['Time_on_Task_s'].append(np.random.lognormal(4, 0.8))
        data['Design_Version'].append(version)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 7: A/B Testing Data
def generate_ab_test_data(size, seed):
    np.random.seed(seed)
    n = size
    
    variants = ['A', 'B']
    devices = ['Desktop', 'Mobile', 'Tablet']
    countries = ['US', 'UK', 'DE', 'FR', 'CA']
    
    data = {
        'User_ID': [f'U{i+1:05d}' for i in range(n)],
        'Variant': np.random.choice(variants, n),
        'Timestamp': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 30), 
                   seconds=np.random.randint(0, 86400)) for _ in range(n)],
        'Device_Type': np.random.choice(devices, n, p=[0.6, 0.35, 0.05]),
        'Country': np.random.choice(countries, n),
    }
    
    # Variant B should have better conversion
    conversion_rate = {'A': 0.05, 'B': 0.08}
    conversions = np.array([np.random.choice([0, 1], p=[1 - conversion_rate[v], conversion_rate[v]]) 
                           for v in data['Variant']])
    data['Conversion'] = conversions.tolist()
    
    # Other metrics
    data['Click_Count'] = (np.random.poisson(3, n) + conversions * 3).tolist()
    data['Page_Load_Time_ms'] = np.random.lognormal(6, 0.6, n).tolist()
    data['Session_Length_s'] = (np.random.lognormal(4, 0.8, n) + conversions * 10).tolist()
    data['Revenue_USD'] = (np.random.exponential(25, n) * conversions).tolist()
    data['Bounce_Rate_%'] = (80 - conversions * 30 + np.random.normal(0, 10, n)).tolist()
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 8: Funnel / Retention Data
def generate_funnel_data(size, seed):
    np.random.seed(seed)
    n = size
    
    sources = ['Organic', 'Paid', 'Referral', 'Direct', 'Social']
    devices = ['Desktop', 'Mobile', 'Tablet']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'User_ID': [f'U{i+1:05d}' for i in range(n)],
        'Session_ID': [f'S{i+1:05d}' for i in range(n)],
        'Referral_Source': np.random.choice(sources, n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'Device_Type': np.random.choice(devices, n),
        'Region': np.random.choice(regions, n),
    }
    
    # Funnel steps (correlated progression)
    step1_prob = 0.95
    step2_prob = 0.75
    step3_prob = 0.50
    step4_prob = 0.30
    
    data['Step_1_Visited'] = np.random.binomial(1, step1_prob, n)
    data['Step_2_Clicked'] = data['Step_1_Visited'] * np.random.binomial(1, step2_prob, n)
    data['Step_3_Submitted'] = data['Step_2_Clicked'] * np.random.binomial(1, step3_prob, n)
    data['Step_4_Confirmed'] = data['Step_3_Submitted'] * np.random.binomial(1, step4_prob, n)
    
    data['Completion'] = data['Step_4_Confirmed']
    
    # Exit step
    data['Exit_Step'] = [1 if not d['Step_1_Visited'] else 
                         (2 if not d['Step_2_Clicked'] else 
                          (3 if not d['Step_3_Submitted'] else
                           (4 if not d['Step_4_Confirmed'] else 0)))
                         for d in [dict(zip(data.keys(), row)) for row in zip(*data.values())]]
    
    # Time between steps
    data['Time_Between_Steps_s'] = np.random.lognormal(3, 0.7, n)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 9: Search / Information Architecture Log Data
def generate_search_data(size, seed):
    np.random.seed(seed)
    n = size
    
    queries = ['login', 'password reset', 'product search', 'checkout', 'help', 
               'account settings', 'billing', 'support', 'return policy']
    
    data = {
        'User_ID': [],
        'Session_ID': [],
        'Query_Text': [],
        'Query_Length': [],
        'Search_Latency_ms': [],
        'Results_Displayed': [],
        'Clicked_Result_Rank': [],
        'Click_Success': [],
        'Reformulation_Count': [],
        'Task_Success': [],
        'Time_to_Completion_s': [],
        'Dwell_Time_on_Result_ms': [],
    }
    
    n_users = n // 5  # ~5 searches per user
    user_ids = [f'U{i+1:05d}' for i in range(n_users)]
    
    for i in range(n):
        user_id = np.random.choice(user_ids)
        session_id = f"{user_id}_S{np.random.randint(1, 3)}"
        query = np.random.choice(queries)
        
        query_length = len(query.split())
        latencies = np.random.lognormal(4, 0.5)
        
        # Determine success
        task_success = np.random.choice([0, 1], p=[0.25, 0.75])
        clicked_rank = np.random.choice([1, 2, 3, 4, 5, None], p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.1])
        click_success = 1 if clicked_rank is not None and clicked_rank <= 3 else 0
        
        reformulation = int(np.random.poisson(0.5)) if not task_success else 0
        
        data['User_ID'].append(user_id)
        data['Session_ID'].append(session_id)
        data['Query_Text'].append(query)
        data['Query_Length'].append(query_length)
        data['Search_Latency_ms'].append(latencies)
        data['Results_Displayed'].append(np.random.choice([10, 20, 50]))
        data['Clicked_Result_Rank'].append(clicked_rank)
        data['Click_Success'].append(click_success)
        data['Reformulation_Count'].append(reformulation)
        data['Task_Success'].append(task_success)
        data['Time_to_Completion_s'].append(np.random.lognormal(3.5, 0.7))
        data['Dwell_Time_on_Result_ms'].append(np.random.lognormal(7, 1) if clicked_rank else np.nan)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.03)
    
    return df

# Dataset 10: Chatbot / Conversational UX Data
def generate_chatbot_data(size, seed):
    np.random.seed(seed)
    n = size
    
    intents = ['order_status', 'return', 'billing', 'technical_support', 'general_info']
    task_types = ['Transactional', 'Informational', 'Support']
    
    data = {
        'User_ID': [],
        'Session_ID': [],
        'Turn_Number': [],
        'User_Message_Text': [],
        'Bot_Response_Text': [],
        'Intent_Label': [],
        'Intent_Confidence': [],
        'Response_Latency_ms': [],
        'Resolution_Success': [],
        'User_Sentiment_Score': [],
        'Escalated_to_Human': [],
        'Task_Type': [],
        'End_of_Session': [],
    }
    
    n_users = n // 8  # ~8 turns per user
    user_ids = [f'U{i+1:05d}' for i in range(n_users)]
    
    for i in range(n):
        user_id = np.random.choice(user_ids)
        session_id = f"{user_id}_S1"
        turn = (i % 8) + 1
        
        intent = np.random.choice(intents)
        task_type = np.random.choice(task_types)
        
        success = np.random.choice([0, 1], p=[0.3, 0.7])
        latency = np.random.lognormal(5.5, 0.6) if success else np.random.lognormal(6.5, 0.8)
        
        sentiment = np.random.normal(0, 1)
        escalate = np.random.choice([0, 1], p=[0.85, 0.15]) if not success else 0
        
        data['User_ID'].append(user_id)
        data['Session_ID'].append(session_id)
        data['Turn_Number'].append(turn)
        data['User_Message_Text'].append(f"Sample user message for {intent}")
        data['Bot_Response_Text'].append(f"Bot response for {intent}")
        data['Intent_Label'].append(intent)
        data['Intent_Confidence'].append(np.random.beta(7, 2))
        data['Response_Latency_ms'].append(latency)
        data['Resolution_Success'].append(success)
        data['User_Sentiment_Score'].append(sentiment)
        data['Escalated_to_Human'].append(escalate)
        data['Task_Type'].append(task_type)
        data['End_of_Session'].append(1 if turn == 8 else 0)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.025)
    
    return df

# Dataset 11: Accessibility Testing Data
def generate_accessibility_data(size, seed):
    np.random.seed(seed)
    n = size
    
    disability_types = ['Visual', 'Motor', 'Cognitive', 'Hearing']
    assistive_tech = ['Screen_Reader', 'Voice_Control', 'Keyboard_Only', 'Magnifier', 'None']
    tasks = ['Form_Submit', 'Navigation', 'Checkout', 'Search']
    
    data = {
        'Participant_ID': [f'P{i+1:04d}' for i in range(n)],
        'Disability_Type': np.random.choice(disability_types, n),
        'Assistive_Tech': [],
        'Task_Name': np.random.choice(tasks, n),
        'Task_Success': [],
        'Completion_Time_s': [],
        'Error_Count': [],
        'Error_Type': [],
        'Keyboard_Presses': [],
        'Cognitive_Load_Score': [],
        'Satisfaction_Score': [],
        'WCAG_Issue_Count': [],
    }
    
    error_types = ['Navigation', 'Labeling', 'Focus', 'Compatibility']
    
    for i in range(n):
        disability = data['Disability_Type'][i]
        
        # Assign assistive tech based on disability
        if disability == 'Visual':
            tech = np.random.choice(['Screen_Reader', 'Magnifier'], p=[0.7, 0.3])
        elif disability == 'Motor':
            tech = np.random.choice(['Voice_Control', 'Keyboard_Only'], p=[0.6, 0.4])
        else:
            tech = 'None'
        
        data['Assistive_Tech'].append(tech)
        
        # Task success (lower for some disabilities)
        success = np.random.choice([0, 1], p=[0.35, 0.65]) if disability != 'None' else np.random.choice([0, 1], p=[0.15, 0.85])
        data['Task_Success'].append(success)
        
        # Completion time (higher for disabilities)
        base_time = np.random.lognormal(4, 0.8)
        comp_time = base_time * 1.5 if disability != 'None' else base_time
        data['Completion_Time_s'].append(comp_time)
        
        # Error count (zero-inflated Poisson)
        error_count = 0 if np.random.random() < 0.3 else int(np.random.poisson(2.5))
        data['Error_Count'].append(error_count)
        data['Error_Type'].append(np.random.choice(error_types) if error_count > 0 else 'None')
        
        # Other metrics
        data['Keyboard_Presses'].append(int(np.random.poisson(15 + (1-success) * 10)))
        data['Cognitive_Load_Score'].append(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.3, 0.3, 0.15]))
        data['Satisfaction_Score'].append(np.random.choice([4, 5, 6, 7], p=[0.1, 0.3, 0.4, 0.2]))
        data['WCAG_Issue_Count'].append(int(np.random.poisson(2)))
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 12: Experimental Cognitive Task Data
def generate_cognitive_task_data(size, seed):
    np.random.seed(seed)
    n = size
    
    conditions = ['Congruent', 'Incongruent', 'Neutral']
    words = ['Red', 'Blue', 'Green', 'Yellow']
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    
    data = {
        'Participant_ID': [],
        'Trial_ID': [],
        'Condition': [],
        'Stimulus_Word': [],
        'Stimulus_Color': [],
        'Response_Key': [],
        'Correct_Response': [],
        'Reaction_Time_ms': [],
        'Accuracy': [],
        'Trial_Order': [],
        'Feedback': [],
        'Age': [],
        'Gender': [],
        'Session_Number': [],
    }
    
    n_participants = n // 48  # ~48 trials per participant
    ages = np.random.normal(35, 12, n_participants).astype(int).clip(18, 65)
    genders = np.random.choice(['M', 'F', 'Other'], n_participants, p=[0.5, 0.48, 0.02])
    
    for i in range(n):
        p_idx = i % n_participants
        p_id = f'P{p_idx+1:04d}'
        trial = (i % 48) + 1
        session = (i // 48) % 3 + 1
        
        condition = np.random.choice(conditions)
        word = np.random.choice(words)
        color = np.random.choice(colors)
        
        # Congruent: word matches color
        if condition == 'Congruent':
            correct_resp = 1
            accuracy = np.random.binomial(1, 0.95)
            rt = np.random.lognormal(5.5, 0.3)
        elif condition == 'Incongruent':
            correct_resp = 0
            accuracy = np.random.binomial(1, 0.65)
            rt = np.random.lognormal(6.5, 0.4)
        else:  # Neutral
            correct_resp = np.random.choice([0, 1])
            accuracy = np.random.binomial(1, 0.85)
            rt = np.random.lognormal(5.8, 0.35)
        
        data['Participant_ID'].append(p_id)
        data['Trial_ID'].append(trial)
        data['Condition'].append(condition)
        data['Stimulus_Word'].append(word)
        data['Stimulus_Color'].append(color)
        data['Response_Key'].append('Space')
        data['Correct_Response'].append(correct_resp)
        data['Reaction_Time_ms'].append(rt)
        data['Accuracy'].append(accuracy)
        data['Trial_Order'].append(trial)
        data['Feedback'].append('None')
        data['Age'].append(ages[p_idx])
        data['Gender'].append(genders[p_idx])
        data['Session_Number'].append(session)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 13: Game Analytics / Player Telemetry
def generate_game_analytics_data(size, seed):
    np.random.seed(seed)
    n = size
    
    game_modes = ['Cooperative', 'Competitive', 'Solo']
    
    data = {
        'Player_ID': [],
        'Session_ID': [],
        'Level_Number': [],
        'Play_Duration_s': [],
        'Actions_Per_Minute': [],
        'Deaths': [],
        'Score': [],
        'XP_Gained': [],
        'Time_in_Menu_s': [],
        'Item_Used': [],
        'Game_Mode': [],
        'Engagement_Score': [],
        'Session_Success': [],
    }
    
    n_players = n // 5  # ~5 sessions per player
    player_ids = [f'Player_{i+1:04d}' for i in range(n_players)]
    
    items = ['Health_Potion', 'Mana_Potion', 'Grenade', 'Shield', 'None']
    
    for i in range(n):
        player_id = np.random.choice(player_ids)
        session_id = f"{player_id}_S{np.random.randint(1, 5)}"
        mode = np.random.choice(game_modes)
        level = np.random.randint(1, 21)
        
        success = np.random.choice([0, 1], p=[0.3, 0.7])
        
        duration = np.random.lognormal(5, 1.2)
        actions = int(np.random.poisson(45))
        deaths = int(np.random.poisson(2))
        score = np.random.lognormal(10, 1)
        xp = np.random.lognormal(8, 1)
        
        engagement = np.random.beta(6, 3) * 100
        
        data['Player_ID'].append(player_id)
        data['Session_ID'].append(session_id)
        data['Level_Number'].append(level)
        data['Play_Duration_s'].append(duration)
        data['Actions_Per_Minute'].append(actions)
        data['Deaths'].append(deaths)
        data['Score'].append(score)
        data['XP_Gained'].append(xp)
        data['Time_in_Menu_s'].append(np.random.lognormal(3, 0.8))
        data['Item_Used'].append(np.random.choice(items))
        data['Game_Mode'].append(mode)
        data['Engagement_Score'].append(engagement)
        data['Session_Success'].append(success)
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.025)
    
    return df

# Dataset 14: Geo / XR / Spatial Interaction Data
def generate_spatial_data(size, seed):
    np.random.seed(seed)
    n = size
    
    devices = ['Oculus_Quest', 'HTC_Vive', 'AR_Glasses']
    gaze_targets = ['UI_Panel', 'Avatar', 'Object_A', 'Object_B', 'Environment']
    interaction_types = ['Gaze', 'Pointer', 'Gesture', 'Voice']
    
    data = {
        'User_ID': [f'U{i+1:04d}' for i in range(n)],
        'Device_Type': np.random.choice(devices, n),
        'Latitude': np.random.uniform(37.0, 38.0, n),
        'Longitude': np.random.uniform(-122.5, -121.5, n),
        'Session_Timestamp': [datetime(2024, 1, 1) + timedelta(seconds=i * 5) for i in range(n)],
        'Gaze_Target': np.random.choice(gaze_targets, n),
        'Interaction_Type': np.random.choice(interaction_types, n),
        'Task_Success': [],
        'Distance_to_Target_m': [],
        'Head_Orientation_XYZ': [],
        'Latency_ms': [],
        'Completion_Time_s': [],
        'Cognitive_Load_Score': [],
    }
    
    for i in range(n):
        device = data['Device_Type'][i]
        
        success = np.random.choice([0, 1], p=[0.25, 0.75])
        distance = np.random.lognormal(0.5, 1)
        latencies = np.random.lognormal(4.5, 0.6)
        
        comp_time = np.random.lognormal(4, 0.7)
        
        # Head orientation as tuple string
        orientation = f"({np.random.uniform(-1, 1):.2f}, {np.random.uniform(-1, 1):.2f}, {np.random.uniform(-1, 1):.2f})"
        
        data['Task_Success'].append(success)
        data['Distance_to_Target_m'].append(distance)
        data['Head_Orientation_XYZ'].append(orientation)
        data['Latency_ms'].append(latencies)
        data['Completion_Time_s'].append(comp_time)
        data['Cognitive_Load_Score'].append(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.35, 0.25, 0.1]))
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 15: Diary / Mixed-Method Qualitative Data
def generate_diary_data(size, seed):
    np.random.seed(seed)
    n = size
    
    prompts = ['Morning_Routine', 'Afternoon_Activity', 'Evening_Reflection', 'Weekly_Summary']
    emotions = ['Positive', 'Neutral', 'Negative', 'Mixed']
    themes = ['Usability', 'Aesthetics', 'Performance', 'Value', 'Emotional_Response']
    devices = ['Mobile', 'Desktop', 'Tablet']
    environments = ['Home', 'Office', 'Transit', 'Other']
    
    sample_responses = [
        "Found the interface intuitive and easy to navigate.",
        "Experienced some lag during peak hours.",
        "The design is visually appealing but functionality needs work.",
        "Overall positive experience, would recommend.",
        "Frustrated with the number of steps required.",
    ]
    
    data = {
        'Participant_ID': [f'P{i+1:04d}' for i in range(n)],
        'Entry_Date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 30)) for _ in range(n)],
        'Prompt': np.random.choice(prompts, n),
        'Response_Text': [],
        'Emotion_Label': np.random.choice(emotions, n),
        'Sentiment_Score': [],
        'Word_Count': [],
        'Theme_Code': [],
        'Device_Used': np.random.choice(devices, n),
        'Environment': np.random.choice(environments, n),
        'Follow_Up_Flag': [],
    }
    
    for i in range(n):
        emotion = data['Emotion_Label'][i]
        
        # Generate sentiment based on emotion
        if emotion == 'Positive':
            sentiment = np.random.normal(0.7, 0.3)
        elif emotion == 'Negative':
            sentiment = np.random.normal(-0.7, 0.3)
        elif emotion == 'Mixed':
            sentiment = np.random.normal(0, 0.5)
        else:
            sentiment = np.random.normal(0, 0.2)
        
        sentiment = np.clip(sentiment, -1, 1)
        
        word_count = int(np.random.lognormal(4, 0.8))
        
        data['Response_Text'].append(np.random.choice(sample_responses))
        data['Sentiment_Score'].append(sentiment)
        data['Word_Count'].append(word_count)
        data['Theme_Code'].append(np.random.choice(themes))
        data['Follow_Up_Flag'].append(np.random.choice([0, 1], p=[0.8, 0.2]))
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.025)
    
    return df

# Dataset 16: Feature Adoption / Time Series Data
def generate_feature_adoption_data(size, seed):
    np.random.seed(seed)
    n = size
    
    features = ['Search', 'Notifications', 'Profile_Edit', 'Export_Data', 'Dark_Mode']
    regions = ['US', 'EU', 'ASIA', 'Other']
    devices = ['Mobile', 'Desktop', 'Tablet']
    
    data = {
        'User_ID': [f'U{i+1:05d}' for i in range(n)],
        'Cohort_Date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 30)) for _ in range(n)],
        'Feature_Name': np.random.choice(features, n),
        'Feature_Used': [],
        'Days_Since_Launch': [],
        'Usage_Count': [],
        'Retention_Day_1': [],
        'Retention_Day_7': [],
        'Retention_Day_30': [],
        'Session_Length_s': [],
        'Engagement_Score': [],
        'Subscription_Status': [],
        'Region': np.random.choice(regions, n),
        'Device_Type': np.random.choice(devices, n),
    }
    
    for i in range(n):
        days = np.random.randint(0, 60)
        
        # Feature used (higher adoption over time)
        used = np.random.choice([0, 1], p=[0.6 - days * 0.01, 0.4 + days * 0.01])
        
        usage_count = int(np.random.poisson(5)) if used else 0
        
        # Retention (decays over time)
        retention_1 = np.random.choice([0, 1], p=[0.3, 0.7])
        retention_7 = retention_1 * np.random.choice([0, 1], p=[0.5, 0.5])
        retention_30 = retention_7 * np.random.choice([0, 1], p=[0.6, 0.4])
        
        engagement = np.random.beta(5, 3) * 100 if used else np.random.beta(2, 5) * 100
        
        data['Feature_Used'].append(used)
        data['Days_Since_Launch'].append(days)
        data['Usage_Count'].append(usage_count)
        data['Retention_Day_1'].append(retention_1)
        data['Retention_Day_7'].append(retention_7)
        data['Retention_Day_30'].append(retention_30)
        data['Session_Length_s'].append(np.random.lognormal(4, 0.8))
        data['Engagement_Score'].append(engagement)
        data['Subscription_Status'].append(np.random.choice([0, 1], p=[0.65, 0.35]))
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df

# Dataset 17: System-Level UX Metrics
def generate_system_ux_metrics_data(size, seed):
    np.random.seed(seed)
    n = size
    
    tasks = ['Login', 'Upload', 'Search', 'Checkout', 'Settings']
    os_types = ['Windows', 'MacOS', 'iOS', 'Android']
    
    data = {
        'User_ID': [f'U{i+1:05d}' for i in range(n)],
        'Session_ID': [f'S{i+1:06d}' for i in range(n)],
        'Task_Name': np.random.choice(tasks, n),
        'Task_Success': [],
        'Time_to_Complete_s': [],
        'Error_Rate_%': [],
        'Satisfaction_Score': [],
        'Cognitive_Load_Score': [],
        'System_Latency_ms': [],
        'Click_Error_Rate': [],
        'Average_Path_Length': [],
        'Help_Accessed': [],
        'Post_Task_Confidence': [],
        'Device_OS': np.random.choice(os_types, n),
        'Timestamp': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90)) for _ in range(n)],
    }
    
    for i in range(n):
        success = np.random.choice([0, 1], p=[0.2, 0.8])
        
        # Correlate metrics
        latency = np.random.lognormal(4, 0.6)
        error_rate = np.random.exponential(3) + (1 - success) * 10
        satisfaction = np.random.normal(6, 1.2) - (1 - success) * 1.5
        cognitive_load = np.random.normal(3, 0.8) + (1 - success) * 1.5
        
        comp_time = np.random.lognormal(3.5, 0.7) + (1 - success) * 5
        
        data['Task_Success'].append(success)
        data['Time_to_Complete_s'].append(comp_time)
        data['Error_Rate_%'].append(np.clip(error_rate, 0, 50))
        data['Satisfaction_Score'].append(np.clip(satisfaction, 1, 7))
        data['Cognitive_Load_Score'].append(np.clip(cognitive_load, 1, 5))
        data['System_Latency_ms'].append(latency)
        data['Click_Error_Rate'].append(np.clip(np.random.exponential(2), 0, 20))
        data['Average_Path_Length'].append(int(np.random.poisson(5)))
        data['Help_Accessed'].append(np.random.choice([0, 1], p=[0.75, 0.25]))
        data['Post_Task_Confidence'].append(np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5]))
    
    df = pd.DataFrame(data)
    df = inject_missing_values(df, 0.02)
    
    return df


# Dataset generation functions mapping
DATASET_GENERATORS = {
    1: generate_survey_data,
    2: generate_usability_test_data,
    3: generate_interaction_log_data,
    4: generate_eye_tracking_data,
    5: generate_physiological_data,
    6: generate_card_sorting_data,
    7: generate_ab_test_data,
    8: generate_funnel_data,
    9: generate_search_data,
    10: generate_chatbot_data,
    11: generate_accessibility_data,
    12: generate_cognitive_task_data,
    13: generate_game_analytics_data,
    14: generate_spatial_data,
    15: generate_diary_data,
    16: generate_feature_adoption_data,
    17: generate_system_ux_metrics_data,
}

# Dataset names
DATASET_NAMES = {
    1: 'survey_questionnaire',
    2: 'usability_testing',
    3: 'interaction_telemetry',
    4: 'eye_tracking',
    5: 'physiological',
    6: 'card_sorting',
    7: 'ab_testing',
    8: 'funnel_retention',
    9: 'search_ia',
    10: 'chatbot_conversational',
    11: 'accessibility',
    12: 'cognitive_task',
    13: 'game_analytics',
    14: 'spatial_xr',
    15: 'diary_qualitative',
    16: 'feature_adoption',
    17: 'system_ux_metrics',
}

# Main generation function
def generate_all_datasets(small_n=50, medium_n=200, large_n=1000):
    """Generate all datasets with small, medium, and large sizes"""
    
    # Create directories
    import os
    for size in ['small', 'medium', 'large']:
        os.makedirs(size, exist_ok=True)
    
    # Generate each dataset
    for dataset_num in range(1, 18):
        name = DATASET_NAMES[dataset_num]
        generator = DATASET_GENERATORS[dataset_num]
        
        print(f"\nGenerating Dataset {dataset_num}: {name}")
        
        for size_name, size_val, seed in [('small', small_n, 42),
                                           ('medium', medium_n, 123),
                                           ('large', large_n, 456)]:
            print(f"  Creating {size_name} version (n={size_val})...")
            
            # Adjust size for log datasets
            if dataset_num in [3, 9, 10]:  # Log-type datasets should have more rows
                size_val = size_val * 10
                
            df = generator(size_val, seed + dataset_num)
            
            filename = f"{size_name}/{name}_{size_name}.csv"
            df.to_csv(filename, index=False)
            print(f"    Saved to {filename}")
    
    print("\n✓ All datasets generated successfully!")

if __name__ == "__main__":
    generate_all_datasets()

print("Dataset generation script ready. Run generate_all_datasets() to create all datasets.")

