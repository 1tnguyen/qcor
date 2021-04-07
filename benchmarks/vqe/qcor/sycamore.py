from qcor import *

@qjit
def sycamore(q : qreg):
  # Begin hz_1_2
  Rz(q[0], -0.78539816339)
  Rx(q[0], 1.57079632679)
  Rz(q[0], 0.78539816339)
  # End hz_1_2
  Rx(q[1], 1.57079632679)
  Rx(q[2], 1.57079632679)
  # Begin hz_1_2
  Rz(q[3], -0.78539816339)
  Rx(q[3], 1.57079632679)
  Rz(q[3], 0.78539816339)
  # End hz_1_2
  Ry(q[4], 1.57079632679)
  # Begin hz_1_2
  Rz(q[5], -0.78539816339)
  Rx(q[5], 1.57079632679)
  Rz(q[5], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[6], -0.78539816339)
  Rx(q[6], 1.57079632679)
  Rz(q[6], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[7], -0.78539816339)
  Rx(q[7], 1.57079632679)
  Rz(q[7], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[8], -0.78539816339)
  Rx(q[8], 1.57079632679)
  Rz(q[8], 0.78539816339)
  # End hz_1_2
  Rx(q[9], 1.57079632679)
  Ry(q[10], 1.57079632679)
  # Begin hz_1_2
  Rz(q[11], -0.78539816339)
  Rx(q[11], 1.57079632679)
  Rz(q[11], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[12], -0.78539816339)
  Rx(q[12], 1.57079632679)
  Rz(q[12], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[13], -0.78539816339)
  Rx(q[13], 1.57079632679)
  Rz(q[13], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[14], -0.78539816339)
  Rx(q[14], 1.57079632679)
  Rz(q[14], 0.78539816339)
  # End hz_1_2
  Ry(q[15], 1.57079632679)
  Rx(q[16], 1.57079632679)
  Rx(q[17], 1.57079632679)
  # Begin hz_1_2
  Rz(q[18], -0.78539816339)
  Rx(q[18], 1.57079632679)
  Rz(q[18], 0.78539816339)
  # End hz_1_2
  Rx(q[19], 1.57079632679)
  Rx(q[20], 1.57079632679)
  # Begin hz_1_2
  Rz(q[21], -0.78539816339)
  Rx(q[21], 1.57079632679)
  Rz(q[21], 0.78539816339)
  # End hz_1_2
  Ry(q[22], 1.57079632679)
  Rx(q[23], 1.57079632679)
  Rx(q[24], 1.57079632679)
  Ry(q[25], 1.57079632679)
  # Begin hz_1_2
  Rz(q[26], -0.78539816339)
  Rx(q[26], 1.57079632679)
  Rz(q[26], 0.78539816339)
  # End hz_1_2
  Rx(q[27], 1.57079632679)
  # Begin hz_1_2
  Rz(q[28], -0.78539816339)
  Rx(q[28], 1.57079632679)
  Rz(q[28], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[29], -0.78539816339)
  Rx(q[29], 1.57079632679)
  Rz(q[29], 0.78539816339)
  # End hz_1_2
  Rx(q[30], 1.57079632679)
  Rx(q[31], 1.57079632679)
  Rx(q[32], 1.57079632679)
  Ry(q[33], 1.57079632679)
  # Begin hz_1_2
  Rz(q[34], -0.78539816339)
  Rx(q[34], 1.57079632679)
  Rz(q[34], 0.78539816339)
  # End hz_1_2
  Ry(q[35], 1.57079632679)
  # Begin hz_1_2
  Rz(q[36], -0.78539816339)
  Rx(q[36], 1.57079632679)
  Rz(q[36], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[37], -0.78539816339)
  Rx(q[37], 1.57079632679)
  Rz(q[37], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[38], -0.78539816339)
  Rx(q[38], 1.57079632679)
  Rz(q[38], 0.78539816339)
  # End hz_1_2
  Ry(q[39], 1.57079632679)
  # Begin hz_1_2
  Rz(q[40], -0.78539816339)
  Rx(q[40], 1.57079632679)
  Rz(q[40], 0.78539816339)
  # End hz_1_2
  Rx(q[41], 1.57079632679)
  Rx(q[42], 1.57079632679)
  Rx(q[43], 1.57079632679)
  # Begin hz_1_2
  Rz(q[44], -0.78539816339)
  Rx(q[44], 1.57079632679)
  Rz(q[44], 0.78539816339)
  # End hz_1_2
  Ry(q[45], 1.57079632679)
  Rx(q[46], 1.57079632679)
  # Begin hz_1_2
  Rz(q[47], -0.78539816339)
  Rx(q[47], 1.57079632679)
  Rz(q[47], 0.78539816339)
  # End hz_1_2
  Ry(q[48], 1.57079632679)
  Ry(q[49], 1.57079632679)
  Ry(q[50], 1.57079632679)
  # Begin hz_1_2
  Rz(q[51], -0.78539816339)
  Rx(q[51], 1.57079632679)
  Rz(q[51], 0.78539816339)
  # End hz_1_2
  Rx(q[52], 1.57079632679)
  Rz(q[1], 2.432656295030221)
  Rz(q[4], -2.2258827283782336)
  Rz(q[3], -2.7293249642089283)
  Rz(q[7], 1.2106965020980647)
  Rz(q[5], -1.1065190593715701)
  Rz(q[9], 1.7892230375390792)
  Rz(q[6], 0.21199582799561975)
  Rz(q[13], -0.1112833809595124)
  Rz(q[8], 2.893794754566797)
  Rz(q[15], -2.954998228479576)
  Rz(q[10], 1.28422274116456)
  Rz(q[17], -1.1227354153033684)
  Rz(q[12], 1.34658993786433)
  Rz(q[21], -1.7818294429115995)
  Rz(q[14], 2.1872907310718426)
  Rz(q[23], -1.961401963214366)
  Rz(q[16], 1.5928715721088071)
  Rz(q[25], -1.5401880072084049)
  Rz(q[18], -2.591169533039924)
  Rz(q[27], 2.612251899130507)
  Rz(q[20], 2.4047981541022003)
  Rz(q[30], -2.394731656273015)
  Rz(q[22], -2.3659320009488227)
  Rz(q[32], 2.2191901639170335)
  Rz(q[24], -2.435037316131151)
  Rz(q[34], 3.0221985839801064)
  Rz(q[26], -2.6108064207399715)
  Rz(q[36], 2.560219630921921)
  Rz(q[29], 1.7858344832951236)
  Rz(q[37], -1.7147844803366754)
  Rz(q[31], 1.4590820918159937)
  Rz(q[39], -2.7148644518627782)
  Rz(q[33], 1.056801934900941)
  Rz(q[41], -1.2947483188429902)
  Rz(q[35], 2.768299098244506)
  Rz(q[43], -2.576048471247718)
  Rz(q[38], -1.6256583901852042)
  Rz(q[44], 1.6003028045332237)
  Rz(q[40], -0.8433460214813505)
  Rz(q[46], 0.8404319695058231)
  Rz(q[42], 2.4767110913273673)
  Rz(q[48], 2.9232217616730027)
  Rz(q[45], 1.7188939810777453)
  Rz(q[49], -1.8507490475427473)
  Rz(q[47], 2.3200371653685714)
  Rz(q[51], -2.3409287132563574)
  Rz(q[50], -0.5750038022731371)
  Rz(q[52], -0.5519833389562016)
  fSim(q[1], q[4], 1.5157741664070026, 0.5567125777724111)
  fSim(q[3], q[7], 1.5177580142210796, 0.49481085782254924)
  fSim(q[5], q[9], 1.603673862122088, 0.47689957001761957)
  fSim(q[6], q[13], 1.517732708964379, 0.5058312223892918)
  fSim(q[8], q[15], 1.52531844440771, 0.46557175536522444)
  fSim(q[10], q[17], 1.6141004604575337, 0.4943434406753406)
  fSim(q[12], q[21], 1.5476810407276227, 0.44290174465705406)
  fSim(q[14], q[23], 1.5237261387831182, 0.46966161228464703)
  fSim(q[16], q[25], 1.52858449420213, 0.5736654641907271)
  fSim(q[18], q[27], 1.5483159975150524, 0.4961408893973949)
  fSim(q[20], q[30], 1.6377079485606105, 0.6888985951517979)
  fSim(q[22], q[32], 1.5299499142361528, 0.4825884757470581)
  fSim(q[24], q[34], 1.5280421758408222, 0.5109767145463228)
  fSim(q[26], q[36], 1.512078286877267, 0.48151528098618757)
  fSim(q[29], q[37], 1.5071938854286824, 0.5089276063739601)
  fSim(q[31], q[39], 1.5460100224552222, 0.5302403303961926)
  fSim(q[33], q[41], 1.516662594039817, 0.45171597904279737)
  fSim(q[35], q[43], 1.4597689731865275, 0.42149859585364335)
  fSim(q[38], q[44], 1.5356494456905732, 0.47076284376184807)
  fSim(q[40], q[46], 1.5179778495709582, 0.5221350266177678)
  fSim(q[42], q[48], 1.4969321270214224, 0.4326117171327447)
  fSim(q[45], q[49], 1.51149872016387, 0.4914319343688027)
  fSim(q[47], q[51], 1.4908807480931237, 0.48862437201319)
  fSim(q[50], q[52], 1.6162569997269376, 0.5014289362839901)
  Rz(q[1], -2.393285439195258)
  Rz(q[4], 2.600059005847245)
  Rz(q[3], 2.129341956555809)
  Rz(q[7], 2.6352148885133277)
  Rz(q[5], 1.0969000068097323)
  Rz(q[9], -0.41419602864222343)
  Rz(q[6], 1.4484708032348872)
  Rz(q[13], -1.3477583561987803)
  Rz(q[8], 2.0866453384245074)
  Rz(q[15], -2.1478488123372865)
  Rz(q[10], 0.08243681179983318)
  Rz(q[17], 0.07905051406135788)
  Rz(q[12], 2.8499779533491796)
  Rz(q[21], 2.9979678487835515)
  Rz(q[14], -1.4497132543866282)
  Rz(q[23], 1.675602022244105)
  Rz(q[16], 1.6110403638231598)
  Rz(q[25], -1.5583567989227558)
  Rz(q[18], 1.8314370494284116)
  Rz(q[27], -1.8103546833378266)
  Rz(q[20], 0.07166413168099722)
  Rz(q[30], -0.06159763385181234)
  Rz(q[22], 2.677585806526769)
  Rz(q[32], -2.8243276435585583)
  Rz(q[24], 2.938289569730543)
  Rz(q[34], -2.3511283018815874)
  Rz(q[26], -2.8965804475722536)
  Rz(q[36], 2.845993657754204)
  Rz(q[29], 2.211436714881895)
  Rz(q[37], -2.1403867119234463)
  Rz(q[31], 2.188142488758683)
  Rz(q[39], 2.839260458374533)
  Rz(q[33], -1.4234721207148076)
  Rz(q[41], 1.1855257367727585)
  Rz(q[35], 1.3200137389449091)
  Rz(q[43], -1.1277631119481197)
  Rz(q[38], 2.9954167840269443)
  Rz(q[44], -3.020772369678924)
  Rz(q[40], 1.6866068204771463)
  Rz(q[46], -1.6895208724526738)
  Rz(q[42], 1.6097339295203592)
  Rz(q[48], -2.49298638369999)
  Rz(q[45], -2.4814450054080606)
  Rz(q[49], 2.3495899389430583)
  Rz(q[47], -2.6928020513609194)
  Rz(q[51], 2.671910503473134)
  Rz(q[50], 2.8571684262235255)
  Rz(q[52], 2.299029739727136)
  Ry(q[0], 1.57079632679)
  Ry(q[1], 1.57079632679)
  Ry(q[2], 1.57079632679)
  Rx(q[3], 1.57079632679)
  Rx(q[4], 1.57079632679)
  Rx(q[5], 1.57079632679)
  Rx(q[6], 1.57079632679)
  Rx(q[7], 1.57079632679)
  Ry(q[8], 1.57079632679)
  # Begin hz_1_2
  Rz(q[9], -0.78539816339)
  Rx(q[9], 1.57079632679)
  Rz(q[9], 0.78539816339)
  # End hz_1_2
  Rx(q[10], 1.57079632679)
  Rx(q[11], 1.57079632679)
  Ry(q[12], 1.57079632679)
  Rx(q[13], 1.57079632679)
  Ry(q[14], 1.57079632679)
  # Begin hz_1_2
  Rz(q[15], -0.78539816339)
  Rx(q[15], 1.57079632679)
  Rz(q[15], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[16], -0.78539816339)
  Rx(q[16], 1.57079632679)
  Rz(q[16], 0.78539816339)
  # End hz_1_2
  Ry(q[17], 1.57079632679)
  Ry(q[18], 1.57079632679)
  # Begin hz_1_2
  Rz(q[19], -0.78539816339)
  Rx(q[19], 1.57079632679)
  Rz(q[19], 0.78539816339)
  # End hz_1_2
  Ry(q[20], 1.57079632679)
  Ry(q[21], 1.57079632679)
  # Begin hz_1_2
  Rz(q[22], -0.78539816339)
  Rx(q[22], 1.57079632679)
  Rz(q[22], 0.78539816339)
  # End hz_1_2
  Ry(q[23], 1.57079632679)
  Ry(q[24], 1.57079632679)
  # Begin hz_1_2
  Rz(q[25], -0.78539816339)
  Rx(q[25], 1.57079632679)
  Rz(q[25], 0.78539816339)
  # End hz_1_2
  Rx(q[26], 1.57079632679)
  # Begin hz_1_2
  Rz(q[27], -0.78539816339)
  Rx(q[27], 1.57079632679)
  Rz(q[27], 0.78539816339)
  # End hz_1_2
  Ry(q[28], 1.57079632679)
  Ry(q[29], 1.57079632679)
  Ry(q[30], 1.57079632679)
  Ry(q[31], 1.57079632679)
  # Begin hz_1_2
  Rz(q[32], -0.78539816339)
  Rx(q[32], 1.57079632679)
  Rz(q[32], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[33], -0.78539816339)
  Rx(q[33], 1.57079632679)
  Rz(q[33], 0.78539816339)
  # End hz_1_2
  Ry(q[34], 1.57079632679)
  Rx(q[35], 1.57079632679)
  Rx(q[36], 1.57079632679)
  Rx(q[37], 1.57079632679)
  Ry(q[38], 1.57079632679)
  Rx(q[39], 1.57079632679)
  Ry(q[40], 1.57079632679)
  # Begin hz_1_2
  Rz(q[41], -0.78539816339)
  Rx(q[41], 1.57079632679)
  Rz(q[41], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[42], -0.78539816339)
  Rx(q[42], 1.57079632679)
  Rz(q[42], 0.78539816339)
  # End hz_1_2
  Ry(q[43], 1.57079632679)
  Ry(q[44], 1.57079632679)
  Rx(q[45], 1.57079632679)
  # Begin hz_1_2
  Rz(q[46], -0.78539816339)
  Rx(q[46], 1.57079632679)
  Rz(q[46], 0.78539816339)
  # End hz_1_2
  Ry(q[47], 1.57079632679)
  Rx(q[48], 1.57079632679)
  Rx(q[49], 1.57079632679)
  Rx(q[50], 1.57079632679)
  Ry(q[51], 1.57079632679)
  # Begin hz_1_2
  Rz(q[52], -0.78539816339)
  Rx(q[52], 1.57079632679)
  Rz(q[52], 0.78539816339)
  # End hz_1_2
  Rz(q[0], 0.22687454249256764)
  Rz(q[3], 0.5571415271072138)
  Rz(q[2], -2.251399467871014)
  Rz(q[6], 2.254638072528774)
  Rz(q[4], -0.4124680006679858)
  Rz(q[8], 0.4133873827301967)
  Rz(q[7], 2.356309967771182)
  Rz(q[14], -0.9989369553992501)
  Rz(q[9], -2.7510095309578926)
  Rz(q[16], -2.930393474388504)
  Rz(q[11], 2.2215976945331413)
  Rz(q[20], -1.9682073694250377)
  Rz(q[13], 2.6009354607667348)
  Rz(q[22], -2.7001293760781864)
  Rz(q[15], 2.5335560654855565)
  Rz(q[24], -2.396910270116918)
  Rz(q[17], -0.7423813429186988)
  Rz(q[26], 0.7991639946126939)
  Rz(q[19], 2.034521864394984)
  Rz(q[29], -2.0509383227378613)
  Rz(q[21], -0.6559509011901735)
  Rz(q[31], -0.8184806876377421)
  Rz(q[23], -2.4433245645002115)
  Rz(q[33], 2.7662935758888514)
  Rz(q[25], -1.6140341496882316)
  Rz(q[35], 1.5135909743354203)
  Rz(q[30], 2.4170911887168236)
  Rz(q[38], -2.412768992191835)
  Rz(q[32], 0.4457433798449744)
  Rz(q[40], -0.42078376332383466)
  Rz(q[34], -3.0038933332709163)
  Rz(q[42], -2.832399347717372)
  Rz(q[39], -0.9953713874379709)
  Rz(q[45], 0.910504771397505)
  Rz(q[41], -1.0166585860488078)
  Rz(q[47], 0.8774840291195655)
  Rz(q[46], 2.7788749929700014)
  Rz(q[50], -2.6830355239765877)
  fSim(q[0], q[3], 1.5192859850646716, 0.493872455729601)
  fSim(q[2], q[6], 1.512057260926093, 0.521171372195776)
  fSim(q[4], q[8], 1.6150962547250483, 0.7269644142795685)
  fSim(q[7], q[14], 1.526918809893355, 0.5036266469194411)
  fSim(q[9], q[16], 1.5179602369003038, 0.49143282377695374)
  fSim(q[11], q[20], 1.5374274830970271, 0.4511520475997095)
  fSim(q[13], q[22], 1.534863666383271, 0.46559889697608003)
  fSim(q[15], q[24], 1.5194586006331345, 0.5068560732281251)
  fSim(q[17], q[26], 1.5819908111895569, 0.559587559655881)
  fSim(q[19], q[29], 1.5070579127772137, 0.45203194373797173)
  fSim(q[21], q[31], 1.5273119702494256, 0.4920204543143481)
  fSim(q[23], q[33], 1.5139592614726292, 0.4591694303587969)
  fSim(q[25], q[35], 1.5425204049672618, 0.5057437054391836)
  fSim(q[30], q[38], 1.5432564804094469, 0.565878029265391)
  fSim(q[32], q[40], 1.461108168061224, 0.5435451345370452)
  fSim(q[34], q[42], 1.4529959991144439, 0.4383081386353485)
  fSim(q[39], q[45], 1.6239619425877152, 0.4985036227887463)
  fSim(q[41], q[47], 1.4884560669187343, 0.4964777432808015)
  fSim(q[46], q[50], 1.5034760541421204, 0.5004774067922128)
  Rz(q[0], 0.3225666454467521)
  Rz(q[3], 0.46144942415302936)
  Rz(q[2], 1.666284586910726)
  Rz(q[6], -1.6630459822529655)
  Rz(q[4], -2.0807726116384373)
  Rz(q[8], 2.0816919937006486)
  Rz(q[7], 0.2131062416001404)
  Rz(q[14], 1.1442667707717913)
  Rz(q[9], -3.057055440423448)
  Rz(q[16], -2.624347564922948)
  Rz(q[11], -1.6842610823271318)
  Rz(q[20], 1.9376514074352345)
  Rz(q[13], -1.9233198015747754)
  Rz(q[22], 1.8241258862633238)
  Rz(q[15], 3.041226064990648)
  Rz(q[24], -2.9045802696220098)
  Rz(q[17], -2.0530048626438666)
  Rz(q[26], 2.1097875143378615)
  Rz(q[19], -2.32705251209826)
  Rz(q[29], 2.3106360537553825)
  Rz(q[21], -0.19919797967157285)
  Rz(q[31], -1.2752336091563428)
  Rz(q[23], -1.4317984852307044)
  Rz(q[33], 1.7547674966193443)
  Rz(q[25], 2.90219248657342)
  Rz(q[35], -3.0026356619262313)
  Rz(q[30], 1.9833150852686972)
  Rz(q[38], -1.97899288874371)
  Rz(q[32], -0.6369338680733281)
  Rz(q[40], 0.6618934845944696)
  Rz(q[34], 1.6098110903737093)
  Rz(q[42], -1.1629184641819974)
  Rz(q[39], -0.9546461682461381)
  Rz(q[45], 0.8697795522056723)
  Rz(q[41], 0.8257442823253324)
  Rz(q[47], -0.9649188392545729)
  Rz(q[46], 2.356987377884869)
  Rz(q[50], -2.2611479088914552)
  Rx(q[0], 1.57079632679)
  Rx(q[1], 1.57079632679)
  Rx(q[2], 1.57079632679)
  # Begin hz_1_2
  Rz(q[3], -0.78539816339)
  Rx(q[3], 1.57079632679)
  Rz(q[3], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[4], -0.78539816339)
  Rx(q[4], 1.57079632679)
  Rz(q[4], 0.78539816339)
  # End hz_1_2
  Ry(q[5], 1.57079632679)
  # Begin hz_1_2
  Rz(q[6], -0.78539816339)
  Rx(q[6], 1.57079632679)
  Rz(q[6], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[7], -0.78539816339)
  Rx(q[7], 1.57079632679)
  Rz(q[7], 0.78539816339)
  # End hz_1_2
  Rx(q[8], 1.57079632679)
  Ry(q[9], 1.57079632679)
  Ry(q[10], 1.57079632679)
  Ry(q[11], 1.57079632679)
  Rx(q[12], 1.57079632679)
  Ry(q[13], 1.57079632679)
  Rx(q[14], 1.57079632679)
  Ry(q[15], 1.57079632679)
  Ry(q[16], 1.57079632679)
  Rx(q[17], 1.57079632679)
  Rx(q[18], 1.57079632679)
  Ry(q[19], 1.57079632679)
  # Begin hz_1_2
  Rz(q[20], -0.78539816339)
  Rx(q[20], 1.57079632679)
  Rz(q[20], 0.78539816339)
  # End hz_1_2
  Rx(q[21], 1.57079632679)
  Rx(q[22], 1.57079632679)
  Rx(q[23], 1.57079632679)
  # Begin hz_1_2
  Rz(q[24], -0.78539816339)
  Rx(q[24], 1.57079632679)
  Rz(q[24], 0.78539816339)
  # End hz_1_2
  Ry(q[25], 1.57079632679)
  Ry(q[26], 1.57079632679)
  Ry(q[27], 1.57079632679)
  # Begin hz_1_2
  Rz(q[28], -0.78539816339)
  Rx(q[28], 1.57079632679)
  Rz(q[28], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[29], -0.78539816339)
  Rx(q[29], 1.57079632679)
  Rz(q[29], 0.78539816339)
  # End hz_1_2
  Rx(q[30], 1.57079632679)
  Rx(q[31], 1.57079632679)
  Rx(q[32], 1.57079632679)
  Rx(q[33], 1.57079632679)
  Rx(q[34], 1.57079632679)
  # Begin hz_1_2
  Rz(q[35], -0.78539816339)
  Rx(q[35], 1.57079632679)
  Rz(q[35], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[36], -0.78539816339)
  Rx(q[36], 1.57079632679)
  Rz(q[36], 0.78539816339)
  # End hz_1_2
  Ry(q[37], 1.57079632679)
  # Begin hz_1_2
  Rz(q[38], -0.78539816339)
  Rx(q[38], 1.57079632679)
  Rz(q[38], 0.78539816339)
  # End hz_1_2
  Ry(q[39], 1.57079632679)
  # Begin hz_1_2
  Rz(q[40], -0.78539816339)
  Rx(q[40], 1.57079632679)
  Rz(q[40], 0.78539816339)
  # End hz_1_2
  Rx(q[41], 1.57079632679)
  Rx(q[42], 1.57079632679)
  # Begin hz_1_2
  Rz(q[43], -0.78539816339)
  Rx(q[43], 1.57079632679)
  Rz(q[43], 0.78539816339)
  # End hz_1_2
  Rx(q[44], 1.57079632679)
  # Begin hz_1_2
  Rz(q[45], -0.78539816339)
  Rx(q[45], 1.57079632679)
  Rz(q[45], 0.78539816339)
  # End hz_1_2
  Ry(q[46], 1.57079632679)
  Rx(q[47], 1.57079632679)
  # Begin hz_1_2
  Rz(q[48], -0.78539816339)
  Rx(q[48], 1.57079632679)
  Rz(q[48], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[49], -0.78539816339)
  Rx(q[49], 1.57079632679)
  Rz(q[49], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[50], -0.78539816339)
  Rx(q[50], 1.57079632679)
  Rz(q[50], 0.78539816339)
  # End hz_1_2
  Rx(q[51], 1.57079632679)
  Ry(q[52], 1.57079632679)
  Rz(q[0], 2.729649801458111)
  Rz(q[1], -1.5903685131854368)
  Rz(q[2], 2.159690705273169)
  Rz(q[3], -2.1446661048107574)
  Rz(q[4], 1.1806041264525482)
  Rz(q[5], -1.0584026479570636)
  Rz(q[7], 1.6228145508184577)
  Rz(q[8], -3.090826498553939)
  Rz(q[9], 2.0475467080711205)
  Rz(q[10], -2.8082407726571805)
  Rz(q[11], 0.0363681235151911)
  Rz(q[12], 0.6267368345704036)
  Rz(q[13], -1.6830777237312988)
  Rz(q[14], 1.8212517829565078)
  Rz(q[15], 1.8899056078947574)
  Rz(q[16], -1.801130503082225)
  Rz(q[17], 1.1912247139997103)
  Rz(q[18], -2.126642695386908)
  Rz(q[19], 1.9658871892349727)
  Rz(q[20], -1.9511085272323947)
  Rz(q[21], -1.419800634718331)
  Rz(q[22], 0.5225414603032792)
  Rz(q[23], -2.711935035123221)
  Rz(q[24], 1.753292842879745)
  Rz(q[25], 2.2701744496022607)
  Rz(q[26], -2.4216314985158904)
  Rz(q[28], -1.9757472607935103)
  Rz(q[29], 2.6907181256251422)
  Rz(q[30], -1.526815655445087)
  Rz(q[31], 1.278774958188955)
  Rz(q[32], 2.3593119641931297)
  Rz(q[33], -2.4207968852903035)
  Rz(q[34], 1.648528470046202)
  Rz(q[35], -0.9914944601367082)
  Rz(q[37], 1.3965268522137515)
  Rz(q[38], -1.3773869100671912)
  Rz(q[39], 0.7250694837307659)
  Rz(q[40], -0.7633051520811774)
  Rz(q[41], -2.178226657777811)
  Rz(q[42], 2.43381058936782)
  Rz(q[44], 1.978387193425802)
  Rz(q[45], -2.042821900141575)
  Rz(q[46], 2.616236186508276)
  Rz(q[47], -2.816811184588291)
  Rz(q[49], 0.3777875775333266)
  Rz(q[50], -0.09226863764679546)
  fSim(q[0], q[1], 1.5508555127617396, 0.48773645023970014)
  fSim(q[2], q[3], 1.4860895179183766, 0.49800223593600595)
  fSim(q[4], q[5], 1.5268891182961801, 0.5146971591949128)
  fSim(q[7], q[8], 1.5004518396934141, 0.5412398915468947)
  fSim(q[9], q[10], 1.5996085979257848, 0.5279139399675542)
  fSim(q[11], q[12], 1.5354845176225267, 0.41898979144047055)
  fSim(q[13], q[14], 1.5458428278889307, 0.5336793424906601)
  fSim(q[15], q[16], 1.5651524165812007, 0.5296573901164207)
  fSim(q[17], q[18], 1.6240366191419937, 0.485161082121796)
  fSim(q[19], q[20], 1.6022614099029169, 0.5001380228896636)
  fSim(q[21], q[22], 1.5749311962390906, 0.5236666378689422)
  fSim(q[23], q[24], 1.523830168421918, 0.47521120348928697)
  fSim(q[25], q[26], 1.5426970250653205, 0.5200449092580905)
  fSim(q[28], q[29], 1.4235475054733011, 0.525384127126685)
  fSim(q[30], q[31], 1.5114710633639936, 0.457880755555279)
  fSim(q[32], q[33], 1.5371762819243995, 0.5674318212304652)
  fSim(q[34], q[35], 1.5104144771689965, 0.44988262527027634)
  fSim(q[37], q[38], 1.4985352129034069, 0.63716467833393)
  fSim(q[39], q[40], 1.5073775911322282, 0.4786982840370735)
  fSim(q[41], q[42], 1.4883608214873882, 0.46458301209230124)
  fSim(q[44], q[45], 1.5400981673598617, 0.5128416009466091)
  fSim(q[46], q[47], 1.586087397042518, 0.47904389394283214)
  fSim(q[49], q[50], 1.5630547528567345, 0.4858935687772679)
  Rz(q[0], -2.0554781273338922)
  Rz(q[1], -3.0884258915734355)
  Rz(q[2], -2.2240485334506728)
  Rz(q[3], 2.2390731339130845)
  Rz(q[4], -1.571795675324744)
  Rz(q[5], 1.6939971538202285)
  Rz(q[7], -1.5534983747824278)
  Rz(q[8], 0.08548642704694695)
  Rz(q[9], 1.8515905006898692)
  Rz(q[10], -2.6122845652759277)
  Rz(q[11], -0.6707597723909371)
  Rz(q[12], 1.3338647304765319)
  Rz(q[13], -0.4627326605980698)
  Rz(q[14], 0.600906719823279)
  Rz(q[15], 0.030858809637426078)
  Rz(q[16], 0.05791629517510647)
  Rz(q[17], -0.6573659838052295)
  Rz(q[18], -0.2780519975819692)
  Rz(q[19], -0.1513198647150251)
  Rz(q[20], 0.166098526717603)
  Rz(q[21], -1.6162879329585083)
  Rz(q[22], 0.7190287585434566)
  Rz(q[23], 2.054384388008432)
  Rz(q[24], -3.0130265802519083)
  Rz(q[25], -1.9181090938835224)
  Rz(q[26], 1.7666520449698921)
  Rz(q[28], 1.4266082874858037)
  Rz(q[29], -0.7116374226541716)
  Rz(q[30], -0.3116575563569991)
  Rz(q[31], 0.0636168591008672)
  Rz(q[32], -2.420231334847746)
  Rz(q[33], 2.3587464137505725)
  Rz(q[34], -2.9214866744164656)
  Rz(q[35], -2.704664622854042)
  Rz(q[37], -2.074440199810421)
  Rz(q[38], 2.093580141956982)
  Rz(q[39], 0.03769063813623923)
  Rz(q[40], -0.075926306486649)
  Rz(q[41], 2.369223237804274)
  Rz(q[42], -2.113639306214264)
  Rz(q[44], 0.6857578577092677)
  Rz(q[45], -0.7501925644250381)
  Rz(q[46], 2.757019570571248)
  Rz(q[47], -2.957594568651263)
  Rz(q[49], 1.5950409783059607)
  Rz(q[50], -1.3095220384194293)
  # Begin hz_1_2
  Rz(q[0], -0.78539816339)
  Rx(q[0], 1.57079632679)
  Rz(q[0], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[1], -0.78539816339)
  Rx(q[1], 1.57079632679)
  Rz(q[1], 0.78539816339)
  # End hz_1_2
  Ry(q[2], 1.57079632679)
  Rx(q[3], 1.57079632679)
  Rx(q[4], 1.57079632679)
  Rx(q[5], 1.57079632679)
  Ry(q[6], 1.57079632679)
  Rx(q[7], 1.57079632679)
  Ry(q[8], 1.57079632679)
  Rx(q[9], 1.57079632679)
  # Begin hz_1_2
  Rz(q[10], -0.78539816339)
  Rx(q[10], 1.57079632679)
  Rz(q[10], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[11], -0.78539816339)
  Rx(q[11], 1.57079632679)
  Rz(q[11], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[12], -0.78539816339)
  Rx(q[12], 1.57079632679)
  Rz(q[12], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[13], -0.78539816339)
  Rx(q[13], 1.57079632679)
  Rz(q[13], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[14], -0.78539816339)
  Rx(q[14], 1.57079632679)
  Rz(q[14], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[15], -0.78539816339)
  Rx(q[15], 1.57079632679)
  Rz(q[15], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[16], -0.78539816339)
  Rx(q[16], 1.57079632679)
  Rz(q[16], 0.78539816339)
  # End hz_1_2
  Ry(q[17], 1.57079632679)
  Ry(q[18], 1.57079632679)
  # Begin hz_1_2
  Rz(q[19], -0.78539816339)
  Rx(q[19], 1.57079632679)
  Rz(q[19], 0.78539816339)
  # End hz_1_2
  Ry(q[20], 1.57079632679)
  # Begin hz_1_2
  Rz(q[21], -0.78539816339)
  Rx(q[21], 1.57079632679)
  Rz(q[21], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[22], -0.78539816339)
  Rx(q[22], 1.57079632679)
  Rz(q[22], 0.78539816339)
  # End hz_1_2
  Ry(q[23], 1.57079632679)
  Ry(q[24], 1.57079632679)
  Rx(q[25], 1.57079632679)
  Rx(q[26], 1.57079632679)
  Rx(q[27], 1.57079632679)
  Ry(q[28], 1.57079632679)
  Rx(q[29], 1.57079632679)
  # Begin hz_1_2
  Rz(q[30], -0.78539816339)
  Rx(q[30], 1.57079632679)
  Rz(q[30], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[31], -0.78539816339)
  Rx(q[31], 1.57079632679)
  Rz(q[31], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[32], -0.78539816339)
  Rx(q[32], 1.57079632679)
  Rz(q[32], 0.78539816339)
  # End hz_1_2
  Ry(q[33], 1.57079632679)
  # Begin hz_1_2
  Rz(q[34], -0.78539816339)
  Rx(q[34], 1.57079632679)
  Rz(q[34], 0.78539816339)
  # End hz_1_2
  Rx(q[35], 1.57079632679)
  Rx(q[36], 1.57079632679)
  Rx(q[37], 1.57079632679)
  Rx(q[38], 1.57079632679)
  # Begin hz_1_2
  Rz(q[39], -0.78539816339)
  Rx(q[39], 1.57079632679)
  Rz(q[39], 0.78539816339)
  # End hz_1_2
  Rx(q[40], 1.57079632679)
  # Begin hz_1_2
  Rz(q[41], -0.78539816339)
  Rx(q[41], 1.57079632679)
  Rz(q[41], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[42], -0.78539816339)
  Rx(q[42], 1.57079632679)
  Rz(q[42], 0.78539816339)
  # End hz_1_2
  Ry(q[43], 1.57079632679)
  # Begin hz_1_2
  Rz(q[44], -0.78539816339)
  Rx(q[44], 1.57079632679)
  Rz(q[44], 0.78539816339)
  # End hz_1_2
  Ry(q[45], 1.57079632679)
  Rx(q[46], 1.57079632679)
  # Begin hz_1_2
  Rz(q[47], -0.78539816339)
  Rx(q[47], 1.57079632679)
  Rz(q[47], 0.78539816339)
  # End hz_1_2
  Ry(q[48], 1.57079632679)
  Rx(q[49], 1.57079632679)
  Ry(q[50], 1.57079632679)
  # Begin hz_1_2
  Rz(q[51], -0.78539816339)
  Rx(q[51], 1.57079632679)
  Rz(q[51], 0.78539816339)
  # End hz_1_2
  Rx(q[52], 1.57079632679)
  Rz(q[3], 1.551477711737861)
  Rz(q[4], -1.471042458257343)
  Rz(q[6], 0.9376872138295922)
  Rz(q[7], -2.505708502040649)
  Rz(q[8], -0.8138828334824119)
  Rz(q[9], 1.4625269352539436)
  Rz(q[12], 1.5465631243550486)
  Rz(q[13], -1.3779143806238763)
  Rz(q[14], -3.0565228366971136)
  Rz(q[15], 3.0197041955370203)
  Rz(q[16], 1.431315017932418)
  Rz(q[17], -1.4458071174912288)
  Rz(q[20], -0.5377106953899861)
  Rz(q[21], -0.2326881725325818)
  Rz(q[22], -0.14916236480552586)
  Rz(q[23], -0.004906331434821322)
  Rz(q[24], 0.7506379057685383)
  Rz(q[25], -0.9003639271373485)
  Rz(q[26], -1.8474993478984303)
  Rz(q[27], 1.998770803525135)
  Rz(q[29], 1.245482758924654)
  Rz(q[30], -1.2148704795940866)
  Rz(q[31], 0.036866379598348806)
  Rz(q[32], 0.0035863783361345246)
  Rz(q[33], -1.8613845453504418)
  Rz(q[34], 2.1274320925129424)
  Rz(q[35], -1.349067736156071)
  Rz(q[36], 1.678536049100972)
  Rz(q[38], -2.525710975644985)
  Rz(q[39], 2.108674897933819)
  Rz(q[40], -2.7599208217353106)
  Rz(q[41], 2.720239561457536)
  Rz(q[42], -0.4183020554137533)
  Rz(q[43], 0.9848848466497735)
  Rz(q[45], -0.4216640251949111)
  Rz(q[46], 0.4442859074467593)
  Rz(q[47], -0.7768895120304912)
  Rz(q[48], -0.5622182800397277)
  Rz(q[50], 2.3485402401145627)
  Rz(q[51], -2.7786467500010215)
  fSim(q[3], q[4], 1.4833042346632328, 0.49309862587591086)
  fSim(q[6], q[7], 1.5160176987077063, 0.49850252902924863)
  fSim(q[8], q[9], 1.533861124916149, 0.5011308712768174)
  fSim(q[12], q[13], 1.4801796891587884, 0.47723222215541583)
  fSim(q[14], q[15], 1.5152260103069815, 0.4979623578703301)
  fSim(q[16], q[17], 1.528492454916508, 0.516084701965781)
  fSim(q[20], q[21], 1.5866128438104745, 0.4757798328093993)
  fSim(q[22], q[23], 1.533329585081722, 0.4498338810530747)
  fSim(q[24], q[25], 1.5158107733608834, 0.46637767187376256)
  fSim(q[26], q[27], 1.5645151084458753, 0.47497942677259414)
  fSim(q[29], q[30], 1.565988178478728, 0.5656290235103996)
  fSim(q[31], q[32], 1.5211879663087977, 0.5056110683638724)
  fSim(q[33], q[34], 1.5083349422213166, 0.49641600818147874)
  fSim(q[35], q[36], 1.5087002777201375, 0.44025777694307144)
  fSim(q[38], q[39], 1.5658333118223398, 0.4726453148334656)
  fSim(q[40], q[41], 1.5219378850866572, 0.5335829954492146)
  fSim(q[42], q[43], 1.5501487671052423, 0.44041175393741866)
  fSim(q[45], q[46], 1.482532514866247, 0.6857341218223936)
  fSim(q[47], q[48], 1.494196367390559, 0.45895108234546045)
  fSim(q[50], q[51], 1.5487430259668784, 0.4467898473638142)
  Rz(q[3], -1.8841634261495785)
  Rz(q[4], 1.9645986796300965)
  Rz(q[6], 2.885627836077594)
  Rz(q[7], 1.829536182891349)
  Rz(q[8], 0.05530092720528632)
  Rz(q[9], 0.5933431745662459)
  Rz(q[12], -1.2922413166612754)
  Rz(q[13], 1.460890060392448)
  Rz(q[14], -2.3332667091617383)
  Rz(q[15], 2.2964480680016472)
  Rz(q[16], -1.4365261121897785)
  Rz(q[17], 1.4220340126309714)
  Rz(q[20], 2.603575828646368)
  Rz(q[21], 2.9092106106110616)
  Rz(q[22], -1.740143837266385)
  Rz(q[23], 1.5860751410260376)
  Rz(q[24], 0.5464640634289402)
  Rz(q[25], -0.6961900847977521)
  Rz(q[26], 0.7545429458085753)
  Rz(q[27], -0.6032714901818703)
  Rz(q[29], -1.1636385149103892)
  Rz(q[30], 1.1942507942409564)
  Rz(q[31], -1.0713303304657447)
  Rz(q[32], 1.111783088400228)
  Rz(q[33], 2.0605780161690554)
  Rz(q[34], -1.7945304690065558)
  Rz(q[35], 1.413295584834741)
  Rz(q[36], -1.08382727188984)
  Rz(q[38], 1.3041394540260558)
  Rz(q[39], -1.7211755317372215)
  Rz(q[40], -1.8989017400047816)
  Rz(q[41], 1.859220479727005)
  Rz(q[42], -1.0625073171878154)
  Rz(q[43], 1.6290901084238358)
  Rz(q[45], 1.2492236358489426)
  Rz(q[46], -1.2266017535970946)
  Rz(q[47], 1.4019785896280688)
  Rz(q[48], -2.7410863816982878)
  Rz(q[50], -2.3814810782489255)
  Rz(q[51], 1.9513745683624666)
  Ry(q[0], 1.57079632679)
  Rx(q[1], 1.57079632679)
  Rx(q[2], 1.57079632679)
  Ry(q[3], 1.57079632679)
  # Begin hz_1_2
  Rz(q[4], -0.78539816339)
  Rx(q[4], 1.57079632679)
  Rz(q[4], 0.78539816339)
  # End hz_1_2
  Ry(q[5], 1.57079632679)
  Rx(q[6], 1.57079632679)
  # Begin hz_1_2
  Rz(q[7], -0.78539816339)
  Rx(q[7], 1.57079632679)
  Rz(q[7], 0.78539816339)
  # End hz_1_2
  Rx(q[8], 1.57079632679)
  # Begin hz_1_2
  Rz(q[9], -0.78539816339)
  Rx(q[9], 1.57079632679)
  Rz(q[9], 0.78539816339)
  # End hz_1_2
  Ry(q[10], 1.57079632679)
  Ry(q[11], 1.57079632679)
  Rx(q[12], 1.57079632679)
  Rx(q[13], 1.57079632679)
  Ry(q[14], 1.57079632679)
  Rx(q[15], 1.57079632679)
  Rx(q[16], 1.57079632679)
  Rx(q[17], 1.57079632679)
  Rx(q[18], 1.57079632679)
  Rx(q[19], 1.57079632679)
  # Begin hz_1_2
  Rz(q[20], -0.78539816339)
  Rx(q[20], 1.57079632679)
  Rz(q[20], 0.78539816339)
  # End hz_1_2
  Rx(q[21], 1.57079632679)
  Rx(q[22], 1.57079632679)
  # Begin hz_1_2
  Rz(q[23], -0.78539816339)
  Rx(q[23], 1.57079632679)
  Rz(q[23], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[24], -0.78539816339)
  Rx(q[24], 1.57079632679)
  Rz(q[24], 0.78539816339)
  # End hz_1_2
  Ry(q[25], 1.57079632679)
  # Begin hz_1_2
  Rz(q[26], -0.78539816339)
  Rx(q[26], 1.57079632679)
  Rz(q[26], 0.78539816339)
  # End hz_1_2
  Ry(q[27], 1.57079632679)
  # Begin hz_1_2
  Rz(q[28], -0.78539816339)
  Rx(q[28], 1.57079632679)
  Rz(q[28], 0.78539816339)
  # End hz_1_2
  Ry(q[29], 1.57079632679)
  Ry(q[30], 1.57079632679)
  Ry(q[31], 1.57079632679)
  Rx(q[32], 1.57079632679)
  Rx(q[33], 1.57079632679)
  Ry(q[34], 1.57079632679)
  # Begin hz_1_2
  Rz(q[35], -0.78539816339)
  Rx(q[35], 1.57079632679)
  Rz(q[35], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[36], -0.78539816339)
  Rx(q[36], 1.57079632679)
  Rz(q[36], 0.78539816339)
  # End hz_1_2
  Ry(q[37], 1.57079632679)
  Ry(q[38], 1.57079632679)
  Rx(q[39], 1.57079632679)
  # Begin hz_1_2
  Rz(q[40], -0.78539816339)
  Rx(q[40], 1.57079632679)
  Rz(q[40], 0.78539816339)
  # End hz_1_2
  Ry(q[41], 1.57079632679)
  Rx(q[42], 1.57079632679)
  # Begin hz_1_2
  Rz(q[43], -0.78539816339)
  Rx(q[43], 1.57079632679)
  Rz(q[43], 0.78539816339)
  # End hz_1_2
  Ry(q[44], 1.57079632679)
  # Begin hz_1_2
  Rz(q[45], -0.78539816339)
  Rx(q[45], 1.57079632679)
  Rz(q[45], 0.78539816339)
  # End hz_1_2
  Ry(q[46], 1.57079632679)
  Ry(q[47], 1.57079632679)
  # Begin hz_1_2
  Rz(q[48], -0.78539816339)
  Rx(q[48], 1.57079632679)
  Rz(q[48], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[49], -0.78539816339)
  Rx(q[49], 1.57079632679)
  Rz(q[49], 0.78539816339)
  # End hz_1_2
  Rx(q[50], 1.57079632679)
  Rx(q[51], 1.57079632679)
  # Begin hz_1_2
  Rz(q[52], -0.78539816339)
  Rx(q[52], 1.57079632679)
  Rz(q[52], 0.78539816339)
  # End hz_1_2
  Rz(q[0], 2.5097383157068784)
  Rz(q[1], -1.3704570274342074)
  Rz(q[2], 2.15340751996602)
  Rz(q[3], -2.138382919503608)
  Rz(q[4], 2.814232306319447)
  Rz(q[5], -2.6920308278239626)
  Rz(q[7], -1.8455037387447293)
  Rz(q[8], 0.37749179100923413)
  Rz(q[9], -0.5222760825654607)
  Rz(q[10], -0.23841798202060158)
  Rz(q[11], -2.294693625448506)
  Rz(q[12], 2.957798583534101)
  Rz(q[13], 0.635417654618098)
  Rz(q[14], -0.49724359539288876)
  Rz(q[15], -1.3647843812245368)
  Rz(q[16], 1.4535594860370684)
  Rz(q[17], -2.5347041731580013)
  Rz(q[18], 1.5992861917708034)
  Rz(q[19], -1.5212806562498526)
  Rz(q[20], 1.5360593182524305)
  Rz(q[21], -0.4836060239484377)
  Rz(q[22], -0.4136531504666123)
  Rz(q[23], 2.8800998882669187)
  Rz(q[24], 2.444443226669607)
  Rz(q[25], 2.9676080186992664)
  Rz(q[26], -3.1190650676128966)
  Rz(q[28], 1.7627479969784963)
  Rz(q[29], -1.047777132146871)
  Rz(q[30], -2.927965978946248)
  Rz(q[31], 2.6799252816901165)
  Rz(q[32], -1.3666169229645828)
  Rz(q[33], 1.3051320018674086)
  Rz(q[34], 2.345962039143002)
  Rz(q[35], -1.6889280292335083)
  Rz(q[37], 0.7053764684240956)
  Rz(q[38], -0.6862365262775353)
  Rz(q[39], 1.6612640945006583)
  Rz(q[40], -1.699499762851067)
  Rz(q[41], 2.0126579421111694)
  Rz(q[42], -1.7570740105211604)
  Rz(q[44], 0.8097147262903803)
  Rz(q[45], -0.8741494330061517)
  Rz(q[46], 2.8487140428738082)
  Rz(q[47], -3.049289040953822)
  Rz(q[49], -1.946990986123224)
  Rz(q[50], 2.232509926009754)
  fSim(q[0], q[1], 1.5508555127617396, 0.48773645023970014)
  fSim(q[2], q[3], 1.4860895179183766, 0.49800223593600595)
  fSim(q[4], q[5], 1.5268891182961801, 0.5146971591949128)
  fSim(q[7], q[8], 1.5004518396934141, 0.5412398915468947)
  fSim(q[9], q[10], 1.5996085979257848, 0.5279139399675542)
  fSim(q[11], q[12], 1.5354845176225267, 0.41898979144047055)
  fSim(q[13], q[14], 1.5458428278889307, 0.5336793424906601)
  fSim(q[15], q[16], 1.5651524165812007, 0.5296573901164207)
  fSim(q[17], q[18], 1.6240366191419937, 0.485161082121796)
  fSim(q[19], q[20], 1.6022614099029169, 0.5001380228896636)
  fSim(q[21], q[22], 1.5749311962390906, 0.5236666378689422)
  fSim(q[23], q[24], 1.523830168421918, 0.47521120348928697)
  fSim(q[25], q[26], 1.5426970250653205, 0.5200449092580905)
  fSim(q[28], q[29], 1.4235475054733011, 0.525384127126685)
  fSim(q[30], q[31], 1.5114710633639936, 0.457880755555279)
  fSim(q[32], q[33], 1.5371762819243995, 0.5674318212304652)
  fSim(q[34], q[35], 1.5104144771689965, 0.44988262527027634)
  fSim(q[37], q[38], 1.4985352129034069, 0.63716467833393)
  fSim(q[39], q[40], 1.5073775911322282, 0.4786982840370735)
  fSim(q[41], q[42], 1.4883608214873882, 0.46458301209230124)
  fSim(q[44], q[45], 1.5400981673598617, 0.5128416009466091)
  fSim(q[46], q[47], 1.586087397042518, 0.47904389394283214)
  fSim(q[49], q[50], 1.5630547528567345, 0.4858935687772679)
  Rz(q[0], -1.8355666415826557)
  Rz(q[1], 2.974847929855333)
  Rz(q[2], -2.2177653481435233)
  Rz(q[3], 2.232789948605935)
  Rz(q[4], 3.077761451988355)
  Rz(q[5], -2.955559973492871)
  Rz(q[7], 1.914819914780738)
  Rz(q[8], 2.900353444663767)
  Rz(q[9], -1.861772015853548)
  Rz(q[10], 1.1010779512674858)
  Rz(q[11], 1.660301976572762)
  Rz(q[12], -0.9971970184871672)
  Rz(q[13], -2.78122803894747)
  Rz(q[14], 2.919402098172679)
  Rz(q[15], -2.997636508423279)
  Rz(q[16], 3.086411613235811)
  Rz(q[17], 3.068562903352482)
  Rz(q[18], 2.27920442244032)
  Rz(q[19], -2.9473373264102)
  Rz(q[20], 2.962115988412778)
  Rz(q[21], -2.5524825437284018)
  Rz(q[22], 1.6552233693133516)
  Rz(q[23], 2.74553477179829)
  Rz(q[24], 2.579008343138234)
  Rz(q[25], -2.6155426629805283)
  Rz(q[26], 2.4640856140668985)
  Rz(q[28], -2.3118869702862024)
  Rz(q[29], 3.026857835117831)
  Rz(q[30], 1.089492767144162)
  Rz(q[31], -1.3375334644002939)
  Rz(q[32], 1.3056975523099659)
  Rz(q[33], -1.3671824734071403)
  Rz(q[34], 2.664265063666734)
  Rz(q[35], -2.0072310537572413)
  Rz(q[37], -1.3832898160207692)
  Rz(q[38], 1.4024297581673295)
  Rz(q[39], -0.898503972633654)
  Rz(q[40], 0.8602683042832452)
  Rz(q[41], -1.821661362084707)
  Rz(q[42], 2.077245293674716)
  Rz(q[44], 1.854430324844692)
  Rz(q[45], -1.918865031560465)
  Rz(q[46], 2.5245417142057174)
  Rz(q[47], -2.7251167122857307)
  Rz(q[49], -2.363365765217489)
  Rz(q[50], 2.6488847051040207)
  Rx(q[0], 1.57079632679)
  Ry(q[1], 1.57079632679)
  # Begin hz_1_2
  Rz(q[2], -0.78539816339)
  Rx(q[2], 1.57079632679)
  Rz(q[2], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[3], -0.78539816339)
  Rx(q[3], 1.57079632679)
  Rz(q[3], 0.78539816339)
  # End hz_1_2
  Ry(q[4], 1.57079632679)
  Rx(q[5], 1.57079632679)
  Ry(q[6], 1.57079632679)
  Rx(q[7], 1.57079632679)
  Ry(q[8], 1.57079632679)
  Ry(q[9], 1.57079632679)
  # Begin hz_1_2
  Rz(q[10], -0.78539816339)
  Rx(q[10], 1.57079632679)
  Rz(q[10], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[11], -0.78539816339)
  Rx(q[11], 1.57079632679)
  Rz(q[11], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[12], -0.78539816339)
  Rx(q[12], 1.57079632679)
  Rz(q[12], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[13], -0.78539816339)
  Rx(q[13], 1.57079632679)
  Rz(q[13], 0.78539816339)
  # End hz_1_2
  Rx(q[14], 1.57079632679)
  Ry(q[15], 1.57079632679)
  Ry(q[16], 1.57079632679)
  # Begin hz_1_2
  Rz(q[17], -0.78539816339)
  Rx(q[17], 1.57079632679)
  Rz(q[17], 0.78539816339)
  # End hz_1_2
  Ry(q[18], 1.57079632679)
  Ry(q[19], 1.57079632679)
  Ry(q[20], 1.57079632679)
  Ry(q[21], 1.57079632679)
  Ry(q[22], 1.57079632679)
  Ry(q[23], 1.57079632679)
  Rx(q[24], 1.57079632679)
  # Begin hz_1_2
  Rz(q[25], -0.78539816339)
  Rx(q[25], 1.57079632679)
  Rz(q[25], 0.78539816339)
  # End hz_1_2
  Ry(q[26], 1.57079632679)
  Rx(q[27], 1.57079632679)
  Ry(q[28], 1.57079632679)
  # Begin hz_1_2
  Rz(q[29], -0.78539816339)
  Rx(q[29], 1.57079632679)
  Rz(q[29], 0.78539816339)
  # End hz_1_2
  Rx(q[30], 1.57079632679)
  # Begin hz_1_2
  Rz(q[31], -0.78539816339)
  Rx(q[31], 1.57079632679)
  Rz(q[31], 0.78539816339)
  # End hz_1_2
  Ry(q[32], 1.57079632679)
  Ry(q[33], 1.57079632679)
  # Begin hz_1_2
  Rz(q[34], -0.78539816339)
  Rx(q[34], 1.57079632679)
  Rz(q[34], 0.78539816339)
  # End hz_1_2
  Rx(q[35], 1.57079632679)
  Ry(q[36], 1.57079632679)
  # Begin hz_1_2
  Rz(q[37], -0.78539816339)
  Rx(q[37], 1.57079632679)
  Rz(q[37], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[38], -0.78539816339)
  Rx(q[38], 1.57079632679)
  Rz(q[38], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[39], -0.78539816339)
  Rx(q[39], 1.57079632679)
  Rz(q[39], 0.78539816339)
  # End hz_1_2
  Rx(q[40], 1.57079632679)
  # Begin hz_1_2
  Rz(q[41], -0.78539816339)
  Rx(q[41], 1.57079632679)
  Rz(q[41], 0.78539816339)
  # End hz_1_2
  # Begin hz_1_2
  Rz(q[42], -0.78539816339)
  Rx(q[42], 1.57079632679)
  Rz(q[42], 0.78539816339)
  # End hz_1_2
  Rx(q[43], 1.57079632679)
  # Begin hz_1_2
  Rz(q[44], -0.78539816339)
  Rx(q[44], 1.57079632679)
  Rz(q[44], 0.78539816339)
  # End hz_1_2
  Rx(q[45], 1.57079632679)
  Rx(q[46], 1.57079632679)
  Rx(q[47], 1.57079632679)
  Rx(q[48], 1.57079632679)
  Rx(q[49], 1.57079632679)
  Ry(q[50], 1.57079632679)
  Ry(q[51], 1.57079632679)
  Ry(q[52], 1.57079632679)

ham = Z(0)

for i in range(1, 53):
  ham *= Z(i)

print(ham)

q = qalloc(53)
# sycamore.print_kernel(q)
import xacc
xacc.set_verbose(True)
obs = sycamore.observe(ham, q)
print(obs)