import pandas as pd
0
sign_language_meta_data = {'id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                                  '011', '012', '013', '014', '015', '016', '017', '018', '019', '010',
                                  '021', '022', '023', '024', '025', '026', '027', '028', '029', '020',
                                  '031', '032', '033', '034', '035', '036', '037', '038', '039', '030',
                                  '041', '042', '043', '044', '045', '046', '047', '048', '049', '040',
                                  '051', '052', '053', '054', '055', '056', '057', '058', '059', '050',
                                  '061', '062', '063', '064'],
                           'description': ['opaque', 'red', 'green', 'yellow', 'bright', 'light_blue', 'colors', 'pink',
                                           'women', 'enemy',
                                           'son', 'man', 'away', 'drawer', 'born', 'learn', 'call', 'skimmer', 'bitter',
                                           'sweet_milk',
                                           'milk', 'water', 'food', 'argentina', 'uruguay', 'country', 'last_name',
                                           'where', 'mock', 'birthday',
                                           'breakfast', 'photo', 'hungry', 'map', 'coin', 'music', 'ship', 'none',
                                           'name', 'patience',
                                           'perfume', 'deaf', 'trap', 'rice', 'barbecue', 'candy', 'chewing_gum',
                                           'spaghetti', 'yogurt', 'accept',
                                           'thanks', 'shut_down', 'appear', 'to_land', 'catch', 'help', 'dance',
                                           'bathe', 'buy', 'copy',
                                           'run', 'realize', 'give', 'find'],
                           'hand': ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'r',
                                    'b', 'b', 'r', 'b', 'b', 'b', 'r', 'r', 'r', 'r',
                                    'r', 'r', 'b', 'b', 'b', 'r', 'r', 'b', 'b', 'b',
                                    'b', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'b',
                                    'b', 'r', 'b', 'r']}

df_sign_lang_meta_data = pd.DataFrame(sign_language_meta_data)

df_sign_lang_meta_data.to_csv('sign_lang_meta_data.csv', index=False)

resolution = '1920x1080'
frame_rate=60
