import random
import sys
import numpy as np
import matplotlib.pyplot as plt

class Kmeans(object):
    def __init__(self, input_data, k):
        self.data = input_data;
        self.k = k;
        #保存聚类中心的索引和类样本的索引
        self.centers = [];
        self.clusters = [];
        self.capacity = len(input_data);
        self.__pick_start_point();
    def __pick_start_point(self):
        indexList = [];
        while len(self.centers) < self.k:
            index = random.randint(0, self.capacity - 1);
            if index not in indexList:
                self.centers.append(self.data[index]);
                indexList.append(index);
    def __distance(self, i, center):
        diff = self.data[i] -center;
        return np.sum(np.power(diff, 2));
    def __calCenter(self, cluster):
        cluster = np.array(cluster);
        return(cluster.T @ np.ones(cluster.shape[0]))/(cluster.shape[0]);
    def cluster(self):
        changed = True;
        while changed:
            self.clusters = [];
            for i in range(self.k):
                self.clusters.append([]);
            for i in range(len(self.data)):
                mindistance = sys.maxsize;
                center = None;
                for j in range(self.k):
                    distance = self.__distance(i, self.centers[j]);
                    if mindistance > distance:
                        mindistance = distance;
                        center = j;
                self.clusters[center].append(self.data[i]);
            new_center = [];
            for i in range(len(self.centers)):
                new_center.append(self.__calCenter(self.clusters[i]).tolist());
            if(np.array(new_center) == self.centers).all():
                changed =False;
            else:
                self.centers = np.array(new_center);

    def plotkmean(cluster):
        xdata = [];
        ydata = [];
        for Cluster in cluster.clusters:
            xsubdata = [];
            ysubdata = [];
            for point in Cluster:
                xsubdata.append(point[0]);
                ysubdata.append(point[1]);
            xdata.append(xsubdata);
            ydata.append(ysubdata);
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
        for i in range(len(xdata)):
            for j in range(len(xdata[i])):
                x = np.array([xdata[i][j], cluster.centers[i][0]]);  #此处array中传入的为一个列表，不要忘记外侧的符号
                y = np.array([ydata[i][j], cluster.centers[i][1]]);
                plt.plot(x, y, color = colors[i],marker = 'o', ms = 7,linestyle='-');
                print(x);
            plt.plot(cluster.centers[i][0], cluster.centers[i][1], color=colors[i], marker='*',ms=20);
            print(x);
            # plt.scatter(cluster.centers[i][0],cluster.centers[i][1],s=350,c='none',alpha = 0.7, linewidth = 1.5, edgecolor =colors[i]);
        plt.grid(True);
        plt.title('K-means');
        plt.show();

if __name__ =='__main__':
    data = [];
    for i in range(20):
        print(i);
        point = [random.randint(1,10),random.randint(1,10)];
        data.append(np.array(point,dtype='float64'));
    cluster = Kmeans(data,3);
    print("qqq");
    cluster.cluster();
    Kmeans.plotkmean(cluster);



