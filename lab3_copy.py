
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt


def draw_figure(set1, set2, dates, min_sim, max_sim, c):
    dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    plt.title('Similarity of opening and closing Bitcoin prices')
    plt.plot(dates, set1, label="open", color="blue", linewidth=0.5)
    plt.plot(dates, set2, label="close", color="green", linewidth=0.5)
    plt.plot(dates[min_sim[0]:min_sim[1]], set1[min_sim[0]:min_sim[1]], label="the best similarity", color="red", linewidth=1)
    plt.plot(dates[min_sim[0]:min_sim[1]], set2[min_sim[0]:min_sim[1]], color="red", linewidth=1)
    plt.plot(dates[max_sim[0]:max_sim[1]], set1[max_sim[0]:max_sim[1]], label="the weakest similarity", color="black", linewidth=1)
    plt.plot(dates[max_sim[0]:max_sim[1]], set2[max_sim[0]:max_sim[1]], color="black", linewidth=1)
    
    plt.xlabel('date')
    plt.ylabel('price in USD')
    plt.legend()
    plt.savefig(f'figure_{c}')
    plt.show()

class Lab:

    csv_file = None
    csv_file_path = ''
    n = 0      # Number of rows

    def __init__(self, file_path):
        self.csv_file_path = file_path
        self.csv_file = pd.read_csv(file_path)
        self.n = len(self.csv_file)
        pass

    def printCSV(self):
        if self.csv_file is None:
            print("Read file with 'readCSVfile' method")
        else:
            print(self.csv_file)
        return

    def choose_sets_lab3(self, p, k, choice, sets_power=0):
        match choice:
            case 1:
                set_name = "Original"
                set1 = self.csv_file['Open']
                set2 = self.csv_file['Close']

            case 2:

                set_name = f"Power{sets_power}"
                set1 = self.csv_file[f'OpenPower{sets_power}']

                # If sets transformed with given power weren't previously calculated
                if len(set1) == 0:
                    print(f"Calculating new set for power: {sets_power}")
                    self.Power_set(sets_power)
                    set1 = self.csv_file[f'OpenPower{sets_power}']
                    set2 = self.csv_file[f'ClosePower{sets_power}']
                else:
                    set2 = self.csv_file[f'ClosePower{sets_power}']

            case 3:
                set_name = "Detrending"

                set1 = self.csv_file['OpenDetr']
                set2 = self.csv_file['CloseDetr']

            case _:
                print("Please enter a number 1, 2 or 3")
                return
        self.Minkowski_standard(p, k, set1, set2)

    def Minkowski_standard(self, p, k, set1, set2):
        # print("Calculating...")
        set_len = len(set1)
        m = set_len / k        # Saving results to the file

        subsets = k*[0]
        subsets_count = k*[0]

        for i in range(set_len):
            ind = math.floor(i/m)

            subsets[ind] += math.pow(abs(set1[i] - set2[i]), p)
            subsets_count[ind] += 1

        for i in range(k):
            subsets[i] = math.pow(subsets[i]/subsets_count[i], 1/p)

        # print(subsets)
        # print(subsets_count)
        # print("Calculations completed!")

        # Saving results (lab3)
        # self.lab3_saving_results(subsets, subsets_count, set_name, p, k, set1, set2)
        return subsets, subsets_count

    def lab4_zad1(self, width, p):
        k = math.floor((width/100)*self.n)
        result_euclidean = []
        print(f"k: {k}")
        print(f"self.n: {self.n}")

        for i in range(k):
            result_euclidean.append(0)

        for i in range(self.n - k):

            set1 = list(self.csv_file['Open'][i:i+k+1].copy())
            set2 = list(self.csv_file['Close'][i:i+k+1].copy())
            # print(f"i: {i}")
            # print(f"set1: {set1}")
            # print(f"set1[0]: {set1[0]}")
            subsets, subsets_count = self.Minkowski_standard(p, 1, set1, set2)
            result_euclidean.append(subsets[0])

        # print(f"result_euclidean: {result_euclidean}\nlen(result_euclidean) = {len(result_euclidean)}")

        self.csv_file[f'MovingWindow{width}'] = result_euclidean
        self.csv_file.to_csv(self.csv_file_path, index=False)

        return

    # Saving results to the file (lab3)
    def lab3_saving_results(self, subsets, subsets_count, set_name, p, k, set1, set2):
        f = open('results', 'a')
        if p == 2:
            f.write(f"Euclidean standard (p = 2) for sets '{set_name}':\n\n")
        else:
            f.write(f"Minkowski standard with p = {p}, for sets '{set_name}':\n\n")

        max_ind = 0
        min_ind = 0
        max_val = 0
        min_val = math.inf
        for i in range(k):
            if subsets[i] > max_val:
                max_val = subsets[i]
                max_ind = i
            if subsets[i] < min_val:
                min_val = subsets[i]
                min_ind = i
            f.write(f"Subset {i}: {subsets[i]}\n")

        min_sum = sum(subsets_count[:min_ind])
        max_sum = sum(subsets_count[:max_ind])

        if min_ind == 0:
            min_begin_ind = 0
            min_end_ind = subsets_count[min_ind] - 1
        else:
            min_begin_ind = min_sum
            min_end_ind = min_sum + subsets_count[min_ind] - 1

        if max_ind == 0:
            max_begin_ind = 0
            max_end_ind = subsets_count[max_ind] - 1
        else:
            max_begin_ind = max_sum
            max_end_ind = max_sum + subsets_count[max_ind] - 1

        f.write(f"\nMin: Subset {min_ind}: {subsets[min_ind]} --> subset_indexes: [{min_begin_ind}, {min_end_ind}]\n"
                f"Max: Subset {max_ind}: {subsets[max_ind]} --> subset_indexes: [{max_begin_ind}, {max_end_ind}]\n"
                f"============================\n\n")
        f.close()

        print("Drawing a figure\nClose window to continue!")
        date = self.csv_file['Date']
        draw_figure(set1.to_numpy().flatten(), set2.to_numpy().flatten(), date.to_numpy().flatten(),  [min_begin_ind, min_end_ind], [max_begin_ind, max_end_ind], choice)

    # Method creating a new pair of columns transformed by powering them with value of 'p'
    def Power_set(self, p):
        trsf_set1 = []
        trsf_set2 = []

        print("Creating new columns...")
        for i in range(self.n):
            trsf_set1.append(self.csv_file['Open'][i]**p)
            trsf_set2.append(self.csv_file['Close'][i]**p)

        self.csv_file[f'OpenPower{p}'] = trsf_set1
        self.csv_file[f'ClosePower{p}'] = trsf_set2
        self.csv_file.to_csv(self.csv_file_path, index=False)
        print(f"New columns created for p = {p}.")
        return

    # Method deleting the pair of existing columns transformed by powering them with value of 'p'
    def Column_del(self, columnName, p=0):
        if p != 0:
            del self.csv_file[f'OpenPower{p}']
            del self.csv_file[f'ClosePower{p}']
        if columnName is not None:
            del self.csv_file[columnName]
        self.csv_file.to_csv(self.csv_file_path, index=False)
        print(f"Deleted columns transformed with power {p}")

    # Method creating a new pair of columns transformed by subtracting trend value from each row value
    def Detrending(self):

        detr_set1 = []  # Column to create and add, de-trended "Open" column
        detr_set2 = []  # Column to create and add, de-trended "Open" column

        x = np.array([(i+1) for i in range(self.n)]).reshape(-1, 1)
        y1 = self.csv_file['Open'].values
        y1 = np.array(y1)
        y2 = self.csv_file['Close'].values
        y2 = np.array(y2)

        reg1 = LinearRegression()  # create object for the class
        reg2 = LinearRegression()  # create object for the class

        reg1.fit(x, y1)  # perform linear regression
        reg2.fit(x, y2)  # perform linear regression

        a1 = reg1.coef_[0]
        b1 = reg1.intercept_
        a2 = reg2.coef_[0]
        b2 = reg2.intercept_

        for i in range(self.n):
            subtract_value1 = (i+1)*a1 + b1
            subtract_value2 = (i+1)*a2 + b2

            detr_set1.append(self.csv_file['Open'][i] - subtract_value1)
            detr_set2.append(self.csv_file['Close'][i] - subtract_value2)

        self.csv_file[f'OpenDetr'] = detr_set1
        self.csv_file[f'CloseDetr'] = detr_set2
        self.csv_file.to_csv(self.csv_file_path, index=False)

        return 0


if __name__ == "__main__":

    power = 5
    sets = 10
    choice = 5  # Choose a set:  1. Original  2. Power  3. Detrending

    # Which transformed-with-power sets should be used
    sets_power = 2

    lab = Lab('BTC-USD_lab4.csv')

    # Create a new pair of columns transformed with power "p"
    # lab.Power_set(p)

    # Delete an existing pair of columns transformed with power "p"
    # lab.Column_del(None, p)

    # Create detrending columns of the original columns
    # lab.Detrending()

    # Calculate Minkowski's standard for previously set 'power' and 'sets' values (lab3)
    # lab.choose_sets_lab3(power, sets, choice)

    # Calculate distance for given moving window width
    width = 5
    lab.lab4_zad1(width, power)

