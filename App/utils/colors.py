import plotly.express as px

def tol_palette():
    colors = ['rgb(120, 28, 129)', 'rgb(82, 27, 128)', 'rgb(68, 47, 139)', 'rgb(63, 76, 159)', 'rgb(64, 105, 180)',
              'rgb(69, 130, 193)', 'rgb(78, 150, 189)', 'rgb(90, 166, 169)', 'rgb(104, 176, 144)', 'rgb(122, 184, 120)',
              'rgb(141, 188, 100)', 'rgb(162, 190, 86)', 'rgb(183, 189, 75)', 'rgb(201, 184, 67)', 'rgb(216, 174, 61)',
              'rgb(226, 158, 55)', 'rgb(231, 134, 50)', 'rgb(230, 103, 45)', 'rgb(225, 68, 39)', 'rgb(217, 33, 32)', 'white']
    
    return colors

def get_colors(size):

    if size <= 8:
        return px.colors.qualitative.Dark2
    
    return tol_palette()