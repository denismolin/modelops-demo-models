aoa feature create-stats-table -m ADLDSD_CHURN.transactions
aoa feature compute-stats -s ADLDSD_CHURN.PIMA -m ADLDSD_CHURN.ADLDSD_CHURN -t continuous -c amount,oldbalanceOrig,newbalanceOrig,oldbalanceDest,newbalanceDest
