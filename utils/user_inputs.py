def get_fire_name():
    while True:
        try:
            fire_name = input('Enter the name for the fire: ').lower()
            if fire_name == '':
                raise ValueError
            break
        except ValueError:
            print('Please enter a valid name.')
    return fire_name


def get_percentage(case=None):
    while True:
        try:
            if case == 'land use':
                prob = input(
                    "Enter sample percentage to use for land cover classification (0-100%): ")
            elif case == 'map':
                prob = input(
                    "Enter the percentage of points to add to the map (0-100%): ")
            prob = prob[:-1] if prob[-1] == '%' else prob
            prob = float(prob) / 100 if float(prob) > 1. else float(prob)
            if 0 < prob <= 1:
                break
            else:
                raise ValueError
        except ValueError:
            print('Please enter a valid number.')
    return prob
