#import pandas as pd
#import numpy as np
#import warnings
#import random
#import category_encoders as ce
#import xgboost as xgb
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score, confusion_matrix
#import pickle
#import json

import joblib
from PD_model_train import TrainValTest, WoeEncode, Model
mod = joblib.load('./mod.pkl')


from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import json
from typing import List, Optional

app = FastAPI()

class LoanData(BaseModel):
    uuid: Optional[str] = None
    default: Optional[int] = None
    account_amount_added_12_24m: Optional[int] = None
    account_days_in_dc_12_24m: Optional[int] = None
    account_days_in_rem_12_24m: Optional[int] = None
    account_days_in_term_12_24m: Optional[int] = None
    account_incoming_debt_vs_paid_0_24m: Optional[float] = None
    account_status: Optional[int] = None
    account_worst_status_0_3m: Optional[int] = None
    account_worst_status_12_24m: Optional[int] = None
    account_worst_status_3_6m: Optional[int] = None
    account_worst_status_6_12m: Optional[int] = None
    age: Optional[int] = None
    avg_payment_span_0_12m: Optional[float] = None
    avg_payment_span_0_3m: Optional[float] = None
    merchant_category: Optional[str] = None
    merchant_group: Optional[str] = None
    has_paid: Optional[bool] = None
    max_paid_inv_0_12m: Optional[int] = None
    max_paid_inv_0_24m: Optional[int] = None
    name_in_email: Optional[str] = None
    num_active_div_by_paid_inv_0_12m: Optional[float] = None
    num_active_inv: Optional[int] = None
    num_arch_dc_0_12m: Optional[int] = None
    num_arch_dc_12_24m: Optional[int] = None
    num_arch_ok_0_12m: Optional[int] = None
    num_arch_ok_12_24m: Optional[int] = None
    num_arch_rem_0_12m: Optional[int] = None
    num_arch_written_off_0_12m: Optional[int] = None
    num_arch_written_off_12_24m: Optional[int] = None
    num_unpaid_bills: Optional[int] = None
    status_last_archived_0_24m: Optional[int] = None
    status_2nd_last_archived_0_24m: Optional[int] = None
    status_3rd_last_archived_0_24m: Optional[int] = None
    status_max_archived_0_6_months: Optional[int] = None
    status_max_archived_0_12_months: Optional[int] = None
    status_max_archived_0_24_months: Optional[int] = None
    recovery_debt: Optional[int] = None
    sum_capital_paid_account_0_12m: Optional[int] = None
    sum_capital_paid_account_12_24m: Optional[int] = None
    sum_paid_inv_0_12m: Optional[int] = None
    time_hours: Optional[float] = None
    worst_status_active_inv: Optional[int] = None

@app.post("/")
async def loans_request(data: List[LoanData]):
    try:
        result = mod.predict(data)

        return(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/loans_file/")
async def loans_file(file: UploadFile):
    try:
        # Read the uploaded JSON file
        json_data = await file.read()
        loan_data_list = json.loads(json_data)

        # Parse and validate each loan item using the LoanData Pydantic model
        loan_data_objects = []
        for item in loan_data_list:
            loan_data_obj = LoanData(**item)
            loan_data_objects.append(loan_data_obj)

        result = mod.predict(loan_data_objects)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)