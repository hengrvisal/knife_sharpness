import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TimeDistributed, Attention, Add, Input, Bidirectional, Dropout, LSTM, Conv1D, MaxPool1D, GlobalAveragePooling1D, BatchNormalization, Dense, Activation, Reshape
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



boning_activity_class_names = ['Idle', 'Walking', 'Steeling', 'Reaching', 'Cutting', 'Dropping']
slicing_activity_class_names = ['Idle', 'Walking', 'Steeling', 'Reaching', 'Cutting', 'Slicing', 'Pulling', 'Placing/Manipulation', 'Dropping']

boning_df_resampled = pd.read_csv('TO_TRAIN/boning_df_resampled.csv')
slicing_df_resampled = pd.read_csv('TO_TRAIN/slicing_df_resampled.csv')


def model_init(time_steps: int, num_classes: int):
    inp = Input(shape=(time_steps,1))

    # first Bi‐LSTM → 256 channels
    x1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x1 = Dropout(0.2)(x1)          # (batch, time_steps, 256)

    # second Bi‐LSTM → 128 channels
    x2 = Bidirectional(LSTM(64, return_sequences=True))(x1)
    x2 = Dropout(0.2)(x2)          # (batch, time_steps, 128)

    # project x2 → 256 channels
    x2_proj = TimeDistributed(Dense(256))(x2)

    # now you can add
    res = Add()([x1, x2_proj])     # (batch, time_steps, 256)

    # self‐attention
    attn = Attention()([res, res])

    # … your Conv blocks, pooling, final Dense …
    c1 = Conv1D(128, 3, padding='same', activation='relu')(attn)
    c1 = BatchNormalization()(c1)
    c1 = MaxPool1D(2)(c1)
    c1 = Dropout(0.2)(c1)

    c2 = Conv1D(256, 3, padding='same', activation='relu')(c1)
    c2 = BatchNormalization()(c2)
    c2 = MaxPool1D(2)(c2)
    c2 = Dropout(0.2)(c2)

    # ── Final pooling & output ───────────────────────────────────────────
    gap = GlobalAveragePooling1D()(c2)
    out = Dense(num_classes, activation='softmax')(gap)

    model = Model(inputs=inp, outputs=out)

    model.summary()
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_on_LSTMCNN(df, *,
                              col='Label',
                              test_size=0.2,
                              val_split=0.2,
                              epochs=150,
                              batch_size=128,
                              patience=4,
                              random_state=42,
                              model_name="unnamed"):
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        stratify=df[col],
        random_state=random_state
    )
    
    # not counting 'Label' and 'sharpness' columns into the time_steps
    time_steps = train_df.shape[1] - 2

    X_train_raw = train_df.drop(columns=['sharpness', 'Label']).values
    y_train_raw = train_df[col].values

    X_test_raw = test_df.drop(columns=['sharpness', 'Label']).values
    y_test_raw = test_df[col].values

    # label encoding
    le = LabelEncoder().fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)

    # scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.fit_transform(X_test_raw)

    # Reshape to 3D for LSTM-CNN model
    time_steps = X_train_scaled.shape[1]
    X_train = X_train_scaled.reshape(-1, time_steps, 1)
    X_test = X_test_scaled.reshape(-1, time_steps, 1)

    print(time_steps, num_classes)

    assert np.isfinite(X_train_scaled).all()
    assert not np.isnan(y_train).any()

    model = model_init(time_steps, num_classes)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # fit with validation split
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stopping, rlr],
        verbose=1
    )

    # evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    model.save(os.path.join("/fred/oz137/visal/models", f"{model_name}.keras"))

    return history, X_test, y_test, test_loss, test_acc


boning_activity_history, boning_activity_X_test, boning_activity_y_test, boning_activity_test_loss, boning_activity_test_acc = train_on_LSTMCNN(
    boning_df_resampled,
    col='Label',
    model_name="boning_activity_recognition_LSTMCNN"
)


slicing_activity_history, slicing_activity_X_test, slicing_activity_y_test, slicing_activity_test_loss, slicing_activity_test_acc = train_on_LSTMCNN(
    slicing_df_resampled,
    col='Label',
    model_name="slicing_activity_recognition_LSTMCNN"
)


boning_sharpness_history, boning_sharpness_X_test, boning_sharpness_y_test, boning_sharpness_test_loss, boning_sharpness_test_acc = train_on_LSTMCNN(
    boning_df_resampled,
    col='sharpness',
    model_name="boning_sharpness_classification_LSTMCNN"
)


slicing_sharpness_history, slicing_sharpness_X_test, slicing_sharpness_y_test, slicing_sharpness_test_loss, slicing_sharpness_test_acc = train_on_LSTMCNN(
    slicing_df_resampled,
    col='sharpness',
    model_name="slicing_sharpness_classification_LSTMCNN"
)
