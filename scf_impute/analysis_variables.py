import pandas as pd
import numpy as np

def get_full_df(dct_data, df_raw_data):
    df_orig_data = dct_data['df_orig_data']

    lst_skip_col = dct_data['lst_skipped_cols']
    df_raw_data[lst_skip_col] = df_orig_data[lst_skip_col]

    return df_raw_data

def get_networth(row):
    farmbus = 0
    heloc = 0
    nh_mort = 0
    x805 = row['x805']
    x808 = row['x808']
    x813 = row['x813']
    x905 = row['x905']
    x908 = row['x908']
    x913 = row['x913']
    x1005 = row['x1005']
    x1008 = row['x1008']
    x1013 = row['x1013']
    x1108 = row['x1108']
    x1109 = row['x1109']
    x1119 = row['x1119']
    x1120 = row['x1120']
    x1130 = row['x1130']
    x1131 = row['x1131']
    x1136 = row['x1136']
    flag781 = 0
    flag782 = 0
    flag67 = 0

    peneq = 0
    ptype1 = [row['x11000'], row['x11100'], row['x11300'], row['x11400']]
    ptype2 = [row['x11001'], row['x11101'], row['x11301'], row['x11401']]
    pamt = [row['x11032'], row['x11132'], row['x11332'], row['x11432']]
    pbor = [row['x11025'], row['x11125'], row['x11325'], row['x11425']]
    pwit = [row['x11031'], row['x11131'], row['x11331'], row['x11431']]
    pall = [row['x11036'], row['x11136'], row['x11336'], row['x11436']]
    ppct = [row['x11037'], row['x11137'], row['x11337'], row['x11437']]
    thrift = 0
    rthrift = 0
    sthrift = 0
    req = 0
    seq = 0

    for i in range(0, len(ptype1)):
        hold = max(0, pamt[i]) * (
        (ptype1[i] == 1) | (ptype2[i] in [2, 3, 4, 6, 20, 21, 22, 26]) | (pbor[i] == 1) | (pwit[i] == 1))
        if (i <= 2):
            rthrift = rthrift + hold
        else:
            sthrift = sthrift + hold
        thrift = thrift + hold
        peneq = peneq + hold * ((pall[i] == 1) + (pall[i] in (3, 30)) * (max(0, ppct[i]) / 10000))
        if (i <= 2):
            req = peneq
        else:
            seq = peneq - req

    hold = 0
    pmop = 0
    if (row['x11259'] > 0):
        if ((ptype1[0] == 1) |
                (ptype1[1] == 1) |
                (ptype2[0] in (2, 3, 4, 6, 20, 21, 22, 26)) |
                (ptype2[1] in (2, 3, 4, 6, 20, 21, 22, 26)) |
                (pwit[0] == 1) |
                (pwit[1] == 1) |
                (pbor[0] == 1) |
                (pbor[1] == 1)):
            pmop = row['x11259']
        elif ((ptype1[0] != 0) & (ptype1[1] != 0) & (pwit[0] != 0) &
                  (pwit[1] != 0)):
            pmop = 0
    else:
        pmop = row['x11259']
    thrift = thrift + pmop
    if (req > 0):
        peneq = peneq + pmop * (req / rthrift)
    else:
        peneq = peneq + pmop / 2
    if (row['x11559'] > 0):
        if ((ptype1[2] == 1) |
                (ptype1[3] == 1) |
                (ptype2[2] in (2, 3, 4, 6, 20, 21, 22, 26)) |
                (ptype2[3] in (2, 3, 4, 6, 20, 21, 22, 26)) |
                (pwit[2] == 10) |
                (pwit[3] == 1) |
                (pbor[2] == 1) |
                (pbor[3] == 1)):
            pmop = row['x11559']
        elif ((ptype1[2] != 0) & (ptype1[3] != 0) & (pwit[2] != 0) &
                  (pwit[3] != 0)):
            pmop = 0
        else:
            pmop = row['x11559']
        thrift = thrift + pmop
        if (seq > 0):
            peneq = peneq + pmop * (seq / sthrift)
        else:
            peneq = peneq + pmop / 2

    call = max(0, row['x3930'])

    saving = max(0, row['x3730'] * (row['x3732'] not in (4, 30))) + \
             max(0, row['x3736'] * (row['x3738'] not in (4, 30))) + \
             max(0, row['x3742'] * (row['x3744'] not in (4, 30))) + \
             max(0, row['x3748'] * (row['x3750'] not in (4, 30))) + \
             max(0, row['x3754'] * (row['x3756'] not in (4, 30))) + \
             max(0, row['x3760'] * (row['x3762'] not in (4, 30))) + \
             max(0, row['x3765'])

    checking = max(0, row['x3506']) * (row['x3507'] == 5) + \
               max(0, row['x3510']) * (row['x3511'] == 5) + \
               max(0, row['x3514']) * (row['x3515'] == 5) + \
               max(0, row['x3518']) * (row['x3519'] == 5) + \
               max(0, row['x3522']) * (row['x3523'] == 5) + \
               max(0, row['x3526']) * (row['x3527'] == 5) + \
               max(0, row['x3529']) * (row['x3527'] == 5)

    mmda = max(0, row['x3506']) * ((row['x3507'] == 1) * (11 <= row['x9113'] <= 13)) + \
           max(0, row['x3510']) * ((row['x3511'] == 1) * (11 <= row['x9114'] <= 13)) + \
           max(0, row['x3514']) * ((row['x3515'] == 1) * (11 <= row['x9115'] <= 13)) + \
           max(0, row['x3518']) * ((row['x3519'] == 1) * (11 <= row['x9116'] <= 13)) + \
           max(0, row['x3522']) * ((row['x3523'] == 1) * (11 <= row['x9117'] <= 13)) + \
           max(0, row['x3526']) * ((row['x3527'] == 1) * (11 <= row['x9118'] <= 13)) + \
           max(0, row['x3529']) * ((row['x3527'] == 1) * (11 <= row['x9118'] <= 13)) + \
           max(0, row['x3730'] * (row['x3732'] in (4, 30)) * ((row['x9259'] >= 11) & (row['x9259'] <= 13))) + \
           max(0, row['x3736'] * (row['x3738'] in (4, 30)) * ((row['x9260'] >= 11) & (row['x9260'] <= 13))) + \
           max(0, row['x3742'] * (row['x3744'] in (4, 30)) * ((row['x9261'] >= 11) & (row['x9261'] <= 13))) + \
           max(0, row['x3748'] * (row['x3750'] in (4, 30)) * ((row['x9262'] >= 11) & (row['x9262'] <= 13))) + \
           max(0, row['x3754'] * (row['x3756'] in (4, 30)) * ((row['x9263'] >= 11) & (row['x9263'] <= 13))) + \
           max(0, row['x3760'] * (row['x3762'] in (4, 30)) * ((row['x9264'] >= 11) & (row['x9264'] <= 13))) + \
           max(0, row['x3765'] * (row['x3762'] in (4, 30)) * ((row['x9264'] >= 11) & (row['x9264'] <= 13)))

    mmmf = max(0, row['x3506']) * (row['x3507'] == 1) * ((row['x9113'] < 11) | (row['x9113'] > 13)) + \
           max(0, row['x3510']) * (row['x3511'] == 1) * ((row['x9114'] < 11) | (row['x9114'] > 13)) + \
           max(0, row['x3514']) * (row['x3515'] == 1) * ((row['x9115'] < 11) | (row['x9115'] > 13)) + \
           max(0, row['x3518']) * (row['x3519'] == 1) * ((row['x9116'] < 11) | (row['x9116'] > 13)) + \
           max(0, row['x3522']) * (row['x3523'] == 1) * ((row['x9117'] < 11) | (row['x9117'] > 13)) + \
           max(0, row['x3526']) * (row['x3527'] == 1) * ((row['x9118'] < 11) | (row['x9118'] > 13)) + \
           max(0, row['x3529']) * (row['x3527'] == 1) * ((row['x9118'] < 11) | (row['x9118'] > 13)) + \
           max(0, row['x3730'] * (row['x3732'] in (4, 30)) * ((row['x9259'] < 11) | (row['x9259'] > 13))) + \
           max(0, row['x3736'] * (row['x3738'] in (4, 30)) * ((row['x9260'] < 11) | (row['x9260'] > 13))) + \
           max(0, row['x3742'] * (row['x3744'] in (4, 30)) * ((row['x9261'] < 11) | (row['x9261'] > 13))) + \
           max(0, row['x3748'] * (row['x3750'] in (4, 30)) * ((row['x9262'] < 11) | (row['x9262'] > 13))) + \
           max(0, row['x3754'] * (row['x3756'] in (4, 30)) * ((row['x9263'] < 11) | (row['x9263'] > 13))) + \
           max(0, row['x3760'] * (row['x3762'] in (4, 30)) * ((row['x9264'] < 11) | (row['x9264'] > 13))) + \
           max(0, row['x3765'] * (row['x3762'] in (4, 30)) * ((row['x9264'] < 11) | (row['x9264'] > 13)))

    cds = max(0, row['x3721'])
    stocks = max(0, row['x3915'])



    mma = mmda + mmmf
    liq = checking + saving + mma + call
    notxbnd = row['x3910']
    mortbnd = row['x3906']
    govtbnd = row['x3908']
    obnd = row['x7634'] + row['x7633']
    bond = notxbnd + mortbnd + govtbnd + obnd
    savbnd = row['x3902']

    stmutf = (row['x3821'] == 1) * max(0, row['x3822'])

    tfbmutf = (row['x3823'] == 1) * max(0, row['x3824'])

    gbmutf = (row['x3825'] == 1) * max(0, row['x3826'])
    obmutf = (row['x3827'] == 1) * max(0, row['x3828'])

    comutf = (row['x3829'] == 1) * max(0, row['x3830'])
    omutf = (row['x7785'] == 1) * max(0, row['x7787'])

    nmmf = stmutf + tfbmutf + gbmutf + obmutf + comutf + omutf

    irakh = row['x6551'] + row['x6559'] + row['x6567'] + row['x6552'] + row['x6560'] + row['x6568'] + row['x6553'] + row['x6561'] + \
                           row['x6569'] + row['x6554'] + row['x6562'] + row['x6570']
    currpen = row['x6462'] + row['x6467'] + row['x6472'] + row['x6477'] + row['x6957']
    futpen = max(0, row['x5604']) + max(0, row['x5612']) + max(0, row['x5620']) + max(0, row['x5628']);

    cashli = max(0, row['x4006'])

    othfin = row['x4018'] + row['x4022'] * (row['x4020'] in (61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 77,
                                                             80, 81, -7)) + row['x4026'] * (
    row['x4024'] in (61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 77, 80, 81, -7)) + row['x4030'] * (
    row['x4028'] in (61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 77, 80, 81, -7))

    othma = max(0, row['x6577']) + max(0, row['x6587'])

    retqliq = irakh + thrift + futpen + currpen

    houses = row['x604'] + row['x614'] + row['x623'] + row['x716'] + ((10000 - max(0, row['x507'])) / 10000) * (
    row['x513'] + row['x526'])

    vehic = max(0, row['x8166']) + max(0, row['x8167']) + max(0, row['x8168']) + max(0, row['x8188']) + max(0, row[
        'x2422']) + max(0, row['x2506']) + max(0, row['x2606']) + max(0, row['x2623'])

    if (row['x507'] > 9000):
        x507 = 9000
        farmbus = 0
        if (x507 > 0):
            farmbus = (x507 / 10000) * (row['x513'] + row['x526'] - row['x805'] - row['x905'] - row['x1005'])
            x805 = row['x805'] * ((10000 - x507) / 10000)
            x808 = row['x808'] * ((10000 - x507) / 10000)
            x813 = row['x813'] * ((10000 - x507) / 10000)
            x905 = row['x905'] * ((10000 - x507) / 10000)
            x908 = row['x908'] * ((10000 - x507) / 10000)
            x913 = row['x913'] * ((10000 - x507) / 10000)
            x1005 = row['x1005'] * ((10000 - x507) / 10000)
            x1008 = row['x1008'] * ((10000 - x507) / 10000)
            x1013 = row['x1013'] * ((10000 - x507) / 10000)
            if (row['x1103'] == 1):
                farmbus = farmbus - row['x1108'] * (x507 / 10000)
                x1108 = row['x1108'] * ((10000 - x507) / 10000)
                x1109 = row['x1109'] * ((10000 - x507) / 10000)
            if (row['x1114'] == 1):
                farmbus = farmbus - row['x1119'] * (x507 / 10000)
                x1119 = row['x1119'] * ((10000 - x507) / 10000)
                x1120 = row['x1120'] * ((10000 - x507) / 10000)
            if (row['x1125'] == 1):
                farmbus = farmbus - row['x1130'] * (x507 / 10000)
                x1130 = row['x1130'] * ((10000 - x507) / 10000)
                x1131 = row['x1131'] * ((10000 - x507) / 10000)
            if (row['x1136'] > 0 & (x1108 + x1119 + x1130 > 0)):
                farmbus = farmbus - row['x1136'] * (x507 / 10000) * ((x1108 * (row['x1103'] == 1)
                                                                      + x1119 * (row['x1114'] == 1) + x1130 * (
                                                                      row['x1125'] == 1)) / (x1108 + x1119 + x1130))
                x1136 = row['x1136'] * ((10000 - x507) / 10000) * (
                (x1108 * (row['x1103'] == 1) + x1119 * (row['x1114'] == 1) + x1130 * (row['x1125'] == 1)) / (
                x1108 + x1119 + x1130))

    if (x1108 + x1119 + x1130) >= 1:
        heloc = x1108 * (row['x1103'] == 1) + x1119 * (row['x1114'] == 1) + x1130 * (row['x1125'] == 1) + max(0,
                                                                                                              x1136) * (
                                                                                                          x1108 * (row[
                                                                                                                       'x1103'] == 1) + x1119 * (
                                                                                                          row[
                                                                                                              'x1114'] == 1) + x1130 * (
                                                                                                          row[
                                                                                                              'x1125'] == 1)) / (
                                                                                                          x1108 + x1119 + x1130)
        mrthel = x805 + x905 + x1005 + x1108 * (row['x1103'] == 1) + x1119 * (row['x1114'] == 1) + x1130 * (
        row['x1125'] == 1) + max(0, x1136) * (
        x1108 * (row['x1103'] == 1) + x1119 * (row['x1114'] == 1) + x1130 * (row['x1125'] == 1)) / (
                             x1108 + x1119 + x1130)
        nh_mort = mrthel - heloc
    else:
        heloc = 0
        mrthel = x805 + x905 + x1005 + .5 * (max(0, x1136)) * (houses > 0)
        nh_mort = mrthel - heloc

    bus = max(0, row['x3129']) + \
          max(0, row['x3124']) - \
          max(0, row['x3126']) * \
          (row['x3127'] == 5) + \
          max(0, row['x3121']) * \
          (row['x3122'] in (1, 6)) + \
          max(0, row['x3229']) + \
          max(0, row['x3224']) - \
          max(0, row['x3226']) * \
          (row['x3227'] == 5) + \
          max(0, row['x3221']) * \
          (row['x3222'] in (1, 6)) + \
          max(0, row['x3335']) + \
          farmbus + max(0, row['x3408']) + \
          max(0, row['x3412']) + \
          max(0, row['x3416']) + \
          max(0, row['x3420']) + \
          max(0, row['x3452']) + \
          max(0, row['x3428'])

    othnfin = row['x4022'] + row['x4026'] + row['x4030'] - othfin + row['x4018']

    oresre = max(row['x1306'], row['x1310']) + \
             max(row['x1325'], row['x1329']) + \
             max(0, row['x1339']) + \
             (row['x1703'] in (12, 14, 21, 22, 25, 40, 41, 42, 43, 44, 49, 50, 52, 999)) * \
             max(0, row['x1706']) * \
             (row['x1705'] / 10000) + \
             (row['x1803'] in (12, 14, 21, 22, 25, 40, 41, 42, 43, 44, 49, 50, 52, 999)) * \
             max(0, row['x1806']) * \
             (row['x1805'] / 10000) + \
             max(0, row['x2002'])

    nnresre = (row['x1703'] in (1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 24, 45, 46, 47, 48, 51, 53, - 7)) * \
              max(0, row['x1706']) * \
              (row['x1705'] / 10000) + \
              (row['x1803'] in (1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 24, 45, 46, 47, 48, 51, 53, - 7)) * \
              max(0, row['x1806']) * \
              (row['x1805'] / 10000) + \
              max(0, row['x2012']) - \
              (row['x1703'] in (1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 24, 45, 46, 47, 48, 51, 53, - 7)) * \
              row['x1715'] * \
              (row['x1705'] / 10000) - \
              (row['x1803'] in (1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 24, 45, 46, 47, 48, 51, 53, - 7)) * \
              row['x1815'] * \
              (row['x1805'] / 10000) - \
              row['x2016']

    if nnresre != 0:
        flag781 = 1
        nnresre = nnresre - row['x2723'] * (row['x2710'] == 78) - row['x2740'] * (row['x2727'] == 78) - row['x2823'] * (
        row['x2810'] == 78) - row['x2840'] * (row['x2827'] == 78) - row['x2923'] * (row['x2910'] == 78) - row['x2940'] * (
                                                                                                          row['x2927'] == 78)

    mort1 = (row['x1703'] in (12, 14, 21, 22, 25, 40, 41, 42, 43, 44, 49, 50, 52, 53, 999)) * row['x1715'] * (row['x1705'] / 10000)
    mort2 = (row['x1803'] in (12, 14, 21, 22, 25, 40, 41, 42, 43, 44, 49, 50, 52, 53, 999)) * row['x1815'] * (row['x1805'] / 10000)
    mort3 = 0

    resdbt = row['x1318'] + row['x1337'] + row['x1342'] + mort1 + mort2 + row['x2006']

    if ((flag781 != 1) & (oresre > 0)):
        flag782 = 1
        resdbt = resdbt + \
                 row['x2723'] * (row['x2710'] == 78) + \
                 row['x2740'] * (row['x2727'] == 78) +\
                 row['x2823'] * (row['x2810'] ==78) +\
                 row['x2840'] * (row['x2827'] == 78) +\
                 row['x2923'] * (row['x2910'] == 78) +\
                 row['x2940'] * (row['x2927'] == 78)

    if (oresre > 0):
        flag67 = 1;
        resdbt = resdbt + \
                 row['x2723'] * (row['x2710'] == 67) + \
                 row['x2740'] * (row['x2727'] == 67) + \
                 row['x2823'] * (row['x2810'] == 67) +\
                 row['x2840'] * (row['x2827'] == 67) + \
                 row['x2923'] * (row['x2910'] == 67)+ \
                 row['x2940'] * (row['x2927'] == 67)

    outpen1 = max(0, row['x11027']) * (row['x11070'] == 5);
    outpen2 = max(0, row['x11127']) * (row['x11170'] == 5);
    outpen4 = max(0, row['x11327']) * (row['x11370'] == 5);
    outpen5 = max(0, row['x11427']) * (row['x11470'] == 5);
    outpen3 = 0;
    outpen6 = 0;
    outmarg = max(0, row['x3932'])

    odebt = outpen1 + outpen2 + outpen4 + outpen5 + max(0, row['x4010']) + max(0, row['x4032']) + outmarg

    ccbal = max(0, row['x427']) + max(0, row['x413']) + max(0, row['x421']) + max(0, row['x430']) + max(0, row['x7575'])

    install = row['x2218'] + row['x2318'] + row['x2418'] + row['x7169'] + row['x2424'] + row['x2519'] + row['x2619'] + row['x2625'] + row['x7183'] + row['x7824'] + row['x7847'] + row['x7870'] + row['x7924'] + row['x7947'] + row['x7970'] + row['x7179'] + row['x1044'] + row['x1215'] + row['x1219']

    othloc = 0
    if (x1108 + x1119 + x1130) >= 1:
        othloc = x1108 * (row['x1103'] != 1) + x1119 * (row['x1114'] != 1) + x1130 * (row['x1125'] != 1) + max(0, x1136) * (x1108 * (row['x1103'] != 1) + x1119 * (row['x1114'] != 1)+
                                                              x1130 * (row['x1125'] != 1)) / (x1108 + x1119 + x1130)


    else:
        othloc = ((houses <= 0) + .5 * (houses > 0)) * (max(0, x1136))

    nfin = vehic + houses + oresre + nnresre + bus + othnfin

    debt = mrthel + resdbt + othloc + ccbal + install + odebt

    fin = liq + cds + nmmf + stocks + bond + retqliq + savbnd + cashli + othma + othfin

    asset = fin + nfin
    networth = asset - debt
    hhsex = row['x8021']
    hdebt = (debt > 0)
    age = row['x14']
    agecl = 1 + (age <= 35) + (age <= 45) + (age <= 55) + (age <= 65) + (age <= 75)

    race = 5
    if row['x6809'] == 1:
        race = 1
    elif row['x6809'] == 2:
        race = 2
    elif row['x6809'] == 3:
        race = 3
    elif row['x6809'] == 4:
        race = 4

    row['checking'] = checking
    row['networth'] = networth
    row['agecl'] = agecl
    row['race'] = race
    row['hhsex'] = hhsex
    row['hdebt'] = hdebt

    return row

def fill_analysis_variables(dct_data, dct_param, df_raw_data):
    df_raw_data = get_full_df(dct_data, df_raw_data)
    df_raw_data = df_raw_data.replace('nan', np.nan)
    df_raw_data = df_raw_data.apply(pd.to_numeric)

    df_raw_data = df_raw_data.apply(lambda row: get_networth(row), axis=1)
    return df_raw_data

