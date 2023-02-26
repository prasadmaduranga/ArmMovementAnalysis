import numpy as np
import pandas as pd
import gsom
import datetime

feature_encode_file = '../data/encoded_features_fclayer_all.csv'
gsom_plot_file="../results/gsom_alldata_plot_train_100_smooth_50_sf_0_86_{}".format(datetime.datetime.now().strftime("%H_%M_%S"))
gsom_output_file= "../results/gsom_out_alldata_train_100_smooth_50_sf_0_86_{}.csv".format(datetime.datetime.now().strftime("%H_%M_%S"))
data_filename = "./zoo.txt".replace('\\', '/')


if __name__ == '__main__':
    # np.random.seed(1)
    # df = pd.read_csv(data_filename)
    # print(df.shape)
    # data_training = df.iloc[:, 1:17]
    # gsom_map = gsom.GSOM(.83, 16, max_radius=4)
    # gsom_map.fit(data_training.to_numpy(), 100, 50)
    # df = df.drop(columns=["label"])
    # map_points = gsom_map.predict(df,"Name")
    # gsom.plot(map_points, "Name", gsom_map=gsom_map)
    # map_points.to_csv("gsom.csv", index=False)

    np.random.seed(1)
    df = pd.read_csv(feature_encode_file)
    print(df.shape)
    data_training = df.iloc[:, :-1]
    gsom_map = gsom.GSOM(.83, 8, max_radius=2)
    # 0.7 is good spread factor value
    gsom_map.fit(data_training.to_numpy(), 100, 50)
    # df = df.drop(columns=["label"])
    map_points = gsom_map.predict(df, "sign")
    gsom.plot(map_points, "sign", gsom_map=gsom_map,file_name=gsom_plot_file)
    map_points.to_csv(gsom_output_file, index=False)

    print("complete")