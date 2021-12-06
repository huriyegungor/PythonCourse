import wbdata

wbdata.search_countries('Turkey') #TUR
wbdata.search_indicators('forest area') #AG.LND.FRST.K2
wbdata.get_data('AG.LND.FRST.K2', country='TUR')[1]

wbdata.search_indicators('life expectancy at birth') #SP.DYN.LE00.IN
wbdata.get_data('SP.DYN.LE00.IN', country='TUR')[1]
wbdata.get_lendingtype()

countries = [i['id'] for i in wbdata.get_country(lendingtype="IBD")]
indicators = {"AG.LND.FRST.K2": "forest area", "SP.DYN.LE00.IN": "life expectancy at birth"}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)

df.to_csv('hw2/wbdata.csv')
print(df.describe())
