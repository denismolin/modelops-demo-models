from teradataml import (
    DataFrame,
    OneHotEncodingFit,
    OneHotEncodingTransform,
    ScaleFit,
    ScaleTransform,
    DecisionForest,
    configure
)
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import numpy as np

configure.val_install_location = 'TRNG_XSP'

def plot_roc_curve(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name   = context.dataset_info.target_names[0]
    entity_key    = context.dataset_info.entity_key

    # read training dataset from Teradata and convert to pandas
    train_df      = DataFrame.from_query(context.dataset_info.sql)
    
    if 'type' in feature_names:
        print ("OneHotEncoding using InDB Functions...")
        
        transaction_types = list(train_df[['type','txn_id']].groupby(['type']).count().to_pandas()['type'].values)


        onehot = OneHotEncodingFit(data           = train_df,
                                        is_input_dense  = True,
                                        target_column      = '"type"',
                                        categorical_values = transaction_types,
                                        other_column="other"
                                       )

        train_df_onehot = OneHotEncodingTransform(data=train_df,
                                           object=onehot.result,
                                           is_input_dense=True
                                          ).result

        onehot.result.to_sql(f"onehot_${context.model_version}", if_exists="replace")
        print("Saved onehot")
        
        feature_names_after_one_hot = [c for c in feature_names if c != 'type'] + ['type_'+c for c in transaction_types]
        category_features = ['type_'+c for c in transaction_types]
    else:
        train_df_onehot = train_df
        feature_names_after_one_hot = feature_names
        category_features = []
    
    print ("Scaling using InDB Functions...")
    
    scaler = ScaleFit(
        data           = train_df_onehot,
        target_columns = feature_names_after_one_hot,
        scale_method   = context.hyperparams["scale_method"],
        miss_value     = context.hyperparams["miss_value"],
        global_scale   = context.hyperparams["global_scale"].lower() in ['true', '1'],
        multiplier     = context.hyperparams["multiplier"],
        intercept      = context.hyperparams["intercept"]
    )

    scaled_train = ScaleTransform(
        data           = train_df_onehot,
        object         = scaler.output,
        accumulate     = [target_name, entity_key]
    ).result
    
    scaler.output.to_sql(f"scaler_${context.model_version}", if_exists="replace")
    print("Saved scaler")
    
    print("Starting training...")
    model = DecisionForest(
        input_columns        = feature_names_after_one_hot,
        response_column      = target_name,
        data                 = scaled_train,
        max_depth            = context.hyperparams["max_depth"],
        num_trees            = context.hyperparams["num_trees"],
        min_node_size        = context.hyperparams["min_node_size"],
        mtry                 = context.hyperparams["mtry"],
        mtry_seed            = context.hyperparams["mtry_seed"],
        seed                 = context.hyperparams["seed"],
        tree_type            = 'CLASSIFICATION'
    )
    
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")    
    print("Saved trained model")

    record_training_stats(
        train_df_onehot,
        features    = feature_names_after_one_hot,
        targets     = [target_name],
        categorical = [target_name]+category_features,
        feature_importance = {f:0 for f in feature_names_after_one_hot},
        context     = context
    )
