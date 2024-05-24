#write a python program to connect to a supabase table and run a select quer
import os
import sys
import json
import requests
import psycopg2
from psycopg2 import sql

# Define your PostgreSQL database connection parameters
db_params = {
    'dbname': 'postgres',
    'user': 'postgres.lnnuafnvfshwlelnimsu',
    'password': 'aaotestcasepassword',
    'host': 'aws-0-us-west-1.pooler.supabase.com',
    'port': '5432'
}

# Define the query embedding, match threshold, and match count
query_embedding = [-0.020826062187552452, -0.018762338906526566, 0.018168851733207703, -0.03364002704620361, -0.021136295050382614, -0.03649956360459328, -0.02191862091422081, 0.01796652562916279, -0.030537698417901993, -0.06598516553640366, -0.028487462550401688, -0.016900943592190742, -0.0234832726418972, 0.0049333758652210236, 0.0354204922914505, 0.06085957959294319, -0.021756760776042938, -0.04235352203249931, 0.01497210469096899, -0.022363737225532532, 0.014284197241067886, -0.015187918208539486, -0.014149312861263752, -0.032264210283756256, -0.02318652905523777, 0.018721874803304672, -0.04577957093715668, 0.03555537387728691, 0.06544563174247742, -0.014999081380665302, -0.04262328892946243, -0.023780018091201782, 0.003560932818800211, -0.03698514401912689, 0.0006895935512147844, -0.014580940827727318, 0.025938158854842186, -0.017170710489153862, -0.02050234191119671, -0.03822607547044754, -0.015201406553387642, -0.013414195738732815, 0.01489117369055748, 0.020731642842292786, 0.04084281995892525, 0.016550244763493538, 0.015794895589351654, -0.041382357478141785, 0.020003270357847214, 0.0029573277570307255, -0.0015326174907386303, -0.008747216314077377, -0.01966606080532074, 0.01586233824491501, 0.014513499103486538, 0.020623736083507538, 0.036472585052251816, -0.0058438414707779884, 0.01939629390835762, -0.011977683752775192, -0.0359870046377182, -0.009374425746500492, 0.01355582382529974, -0.017224663868546486, -0.06447447091341019, 0.046885617077350616, -0.021271178498864174, 0.05381864681839943, -0.040680959820747375, -0.01824978180229664, -0.01586233824491501, -0.028838161379098892, -0.011411171406507492, 0.04807259514927864, -0.04381026700139046, -0.007142098620533943, 0.04000654071569443, 0.03221025690436363, -0.03137397766113281, -0.017319083213806152, 0.007101633120328188, -0.004538840614259243, 0.008180703967809677, 0.0331544429063797, 0.04297398775815964, -0.006056283600628376, -0.06544563174247742, 0.001015000743791461, -0.019423270598053932, 0.013333265669643879, -0.04677771031856537, -0.0037700028624385595, 0.030160022899508476, 0.006973493844270706, 0.030133046209812164, -0.003702560905367136, 0.014567452482879162, 0.001170117175206542, 0.00772884301841259, 0.06323353946208954, 0.04140933230519295, -0.020461875945329666, 0.04998794570565224, -0.018168851733207703, -0.024063274264335632, 0.03374793380498886, -0.03523165360093117, -0.011438148096203804, 0.011923730373382568, -0.021325131878256798, -0.04381026700139046, 0.0007688378100283444, -0.046993523836135864, -0.032264210283756256, 0.003163025714457035, 0.01783164218068123, 0.09161309152841568, -0.04766794294118881, 0.007674889639019966, -0.01124931126832962, -0.027786066755652428, 0.012463265098631382, -0.019720014184713364, 0.0007068755221553147, 0.006245120894163847, 0.0061270976439118385, -0.028055835515260696, 0.04116654396057129, -0.006106865126639605, -0.03709305077791214, -0.016105128452181816, -0.023833971470594406, 0.003351863007992506, -0.0447004996240139, 0.020866528153419495, 0.03890049457550049, 0.01798001304268837, -0.006467679515480995, 0.017818152904510498, 0.04791073501110077, 0.028271649032831192, -0.0014828790444880724, 0.009603728540241718, 0.006781284231692553, -0.01757536269724369, -0.030699558556079865, 0.003134362865239382, 0.04381026700139046, -0.02797490544617176, -0.010419775731861591, -0.028865138068795204, 0.01588931493461132, 0.03185955807566643, 0.03498886525630951, -0.04383724182844162, -0.041921891272068024, -0.019989782944321632, -0.018465595319867134, -0.004882794339209795, 7.198633738880744e-06, 0.037443749606609344, 0.007317447569221258, -0.038172122091054916, -0.009394657798111439, -0.0007102476083673537, -0.0051019806414842606, -0.019059084355831146, 0.025897694751620293, -0.0014567453181371093, 0.009617216885089874, -0.012996056117117405, 0.0373358428478241, 0.023402342572808266, 0.012011404149234295, -0.03264188393950462, -0.05128283053636551, 0.049394454807043076, 0.057406555861234665, -0.03299258276820183, 0.03064560517668724, -0.006700353696942329, 0.06690237671136856, 0.07235167920589447, -0.03693119063973427, 0.010864892043173313, -0.03890049457550049, -0.0027398276142776012, 0.000340581638738513, 0.015309314243495464, 0.017184199765324593, 0.03765956312417984, -0.01994931697845459, -0.005840469617396593, 0.025749322026968002, -0.021864667534828186, -0.0002697676536627114, -0.007425354328006506, -0.009239542298018932, -0.016482803970575333, 0.029836300760507584, 0.05859353393316269, -0.021122805774211884, 0.04035723954439163, -0.03121211752295494, -0.013232102617621422, -0.022255830466747284, 0.008652796968817711, -0.01144489273428917, 0.03482700139284134, -0.006518260575830936, 0.07990517467260361, 0.03952096030116081, 0.007148842792958021, -0.04691259190440178, -0.05522143840789795, -0.03048374317586422, 0.07089493423700333, 0.02260652929544449, 0.024400483816862106, 0.0015486348420381546, 0.018897224217653275, 0.021540947258472443, 0.049826085567474365, -0.004923259373754263, 0.06965400278568268, -0.003437851322814822, 0.004215119406580925, -0.0027431997004896402, -0.04211072996258736, -0.02615397237241268, 0.00021296890918165445, -0.010203961282968521, -0.055733997374773026, 0.049529340118169785, -0.02628885768353939, -0.005034538917243481, -0.04459259286522865, 0.04089677333831787, 0.010493961162865162, -0.017332570627331734, -0.07861029356718063, 0.01004884485155344, 0.056003764271736145, 0.005321166943758726, 0.03256095573306084, 0.0066598886623978615, 0.0034142467193305492, 0.01116838026791811, -0.05214608460664749, 0.04915166646242142, -0.010777217335999012, 0.03161676973104477, -0.007162331137806177, 0.018155362457036972, -0.0044410498812794685, 0.045590732246637344, 0.050662364810705185, 0.028919091448187828, -0.001325233606621623, -0.022228853777050972, 0.017062803730368614, 0.010325356386601925, -0.023442808538675308, 0.002942153485491872, -0.0393860749900341, -0.009745356626808643, -0.038172122091054916, 0.026949787512421608, 0.03644561022520065, -0.012260939925909042, -0.021041875705122948, 0.03207537159323692, 0.04845026880502701, 0.022282807156443596, -0.046858638525009155, 0.030025139451026917, 0.06280190497636795, 0.022876296192407608, -0.009118146263062954, 0.010008379817008972, -0.02951258048415184, -0.04337863624095917, 0.012820707634091377, 0.008416750468313694, 8.382818487007171e-05, 0.00836954079568386, 0.006602562963962555, -0.08648750931024551, 0.017750710248947144, 0.00912489090114832, 0.023213505744934082, -0.028838161379098892, 0.024940019473433495, -0.041085612028837204, -0.01672559417784214, 0.002254245802760124, 0.03223723545670509, -0.05152561888098717, -0.011485357768833637, -0.01657722145318985, 0.033774908632040024, 0.043324682861566544, 0.02864932455122471, 0.042083751410245895, -0.012530706822872162, 0.024899553507566452, -0.015835361555218697, 0.0051087248139083385, 0.004420817364007235, -0.01412233617156744, -0.022417690604925156, 0.04132840409874916, 0.020313503220677376, 0.0458335243165493, 0.007614191621541977, -0.04666980355978012, -0.03285769745707512, 0.04677771031856537, 0.028757231310009956, -0.09333960711956024, 0.0053380271419882774, -0.01320512592792511, -0.05087817832827568, 0.022687459364533424, -0.008598843589425087, -0.01769675686955452, -0.00821442436426878, -0.019908852875232697, 0.01418977789580822, -0.011997915804386139, -0.01673908159136772, -0.06808935105800629, -0.0373358428478241, -0.05174143612384796, 0.0059719812124967575, 0.013414195738732815, -0.05686701834201813, 0.02585722878575325, 0.012267683632671833, -0.01657722145318985, -0.007007214706391096, -0.0064980280585587025, -0.04739817604422569, -0.019841410219669342, -0.025479553267359734, 0.012200241908431053, -0.009212564677000046, -0.06123725697398186, 0.010588380508124828, 0.028217695653438568, 0.0454828254878521, -0.01842512935400009, 0.005054771434515715, 0.03593305125832558, -0.0028645952697843313, -0.02079908549785614, 0.039844680577516556, 0.008093029260635376, -0.031427931040525436, 0.03917026147246361, -0.01714373379945755, -0.04901678115129471, -0.0234832726418972, 0.03873863443732262, -0.018479084596037865, 0.00900349486619234, -0.012382335029542446, 0.02514234371483326, -0.014041406102478504, 0.020016759634017944, 0.00026133740902878344, 0.03415258228778839, 0.010864892043173313, 0.03188653662800789, -0.051363758742809296, 0.0012982568005099893, -0.04410700872540474, -0.037308864295482635, -0.029539557173848152, 0.0195176899433136, -0.019558154046535492, 0.020273039117455482, 0.011215589940547943, -0.012382335029542446, -0.03334328159689903, 0.010574892163276672, -0.016415361315011978, 0.013960476033389568, 0.03158979117870331, 0.06085957959294319, 0.039682820439338684, -0.016091639176011086, 0.0302139762789011, 0.026248391717672348, 0.018236292526125908, 0.00709488894790411, 0.006973493844270706, 0.10186426341533661, -0.030672581866383553, -0.007877214811742306, 0.04256933555006981, -0.034611187875270844, 0.016388384625315666, 0.03776746988296509, -0.0066194236278533936, -0.020138155668973923, 0.021271178498864174, 0.0034631420858204365, 0.016509780660271645, 0.008275122381746769, -0.013259080238640308, -0.023550715297460556, 0.035636305809020996, 0.05870144069194794, 0.0556800402700901, 0.005853957962244749, -0.003130990779027343, -0.015767918899655342, 0.015363267622888088, 0.0320214182138443, -0.00569209735840559, -0.017507920041680336, -0.006029306910932064, -0.0032473281025886536, 0.019032107666134834, 0.010291635990142822, -0.0029016882181167603, 0.008538145571947098, -0.002456571673974395, 0.02628885768353939, -0.010635589249432087, 0.009360937401652336, -0.049529340118169785, -0.0004944335087202489, -0.050392597913742065, -0.04399910196661949, 0.007526517380028963, 0.03763258829712868, 0.04202979803085327, 0.04399910196661949, -0.02752978913486004, -0.033100489526987076, -0.0354204922914505, -0.04113956540822983, 0.017926059663295746, -0.0032979093957692385, 0.028433509171009064, -0.014176289550960064, -0.008659541606903076, -0.009812798351049423, 0.0013732858933508396, -0.02011117711663246, -0.03285769745707512, -0.0010343902977183461, 0.04019537940621376, -0.02530420571565628, 0.03952096030116081, -0.0018698270432651043, -0.04386422038078308, -0.026518160477280617, -0.053737714886665344, -0.04707445576786995, -0.03582514449954033, -0.06264004856348038, -0.031535837799310684, 0.013973964378237724, 0.017197687178850174, -0.020583271980285645, -0.03204839676618576, -0.01656373403966427, -0.012470009736716747, -0.022255830466747284, 0.000806773838121444, 0.0061203534714877605, -0.005034538917243481, -0.0041038403287529945, 0.031427931040525436, 0.01104024052619934, -0.028028858825564384, -0.05217306315898895, -0.009145122952759266, 0.009536285884678364, 0.00814023893326521, -0.02556048519909382, -0.018748851493000984, -0.021540947258472443, -0.007634424604475498, -0.011013263836503029, -0.004215119406580925, -0.007270237896591425, 0.01630745455622673, 0.021190248429775238, -0.00017650812515057623, -0.014918150380253792, 0.01673908159136772, -0.01376489456743002, -0.00238744355738163, -0.02951258048415184, -0.04224561154842377, -0.030861418694257736, -0.003527211956679821, -0.039682820439338684, -0.03231816366314888, -0.002323373919352889, 0.02854141779243946, -0.04200282320380211, -0.009097914211452007, -0.011175124906003475, 0.029485603794455528, -0.02106885239481926, 0.014135824516415596, -0.00018978575826622546, -0.027732113376259804, -0.03838793560862541, -0.010089309886097908, 0.007850238122045994, -0.015484662726521492, 0.009151867590844631, -0.019153503701090813, 0.03274979069828987, -0.005496515892446041, 0.01560605876147747, -0.009819542057812214, -0.009522797539830208, 0.02403629757463932, 0.0017206118209287524, 0.017197687178850174, -0.02009768970310688, 0.01684699021279812, 0.022107457742094994, 0.07262144982814789, -0.0002191862149629742, 0.003837444819509983, -0.007971634157001972, 0.023537227883934975, 0.026329321786761284, 0.014958616346120834, 0.03992561250925064, -0.0036317468620836735, -0.0024801762774586678, -0.0038003516383469105, 0.03868468105792999, 0.007256749551743269, 0.005199771374464035, 0.009819542057812214, 0.012672334909439087, -0.0031579674687236547, -0.022390713915228844, 0.03342420980334282, -0.009650937281548977, -0.012679079547524452, 0.0054594227112829685, 0.0007018173928372562, 0.019733503460884094, -0.004977213218808174, -0.022660482674837112, 0.037605609744787216, -0.008747216314077377, -0.005820237100124359, 8.53561723488383e-05, -0.0008139395504258573, -0.006804889068007469, 0.018047455698251724, -0.032102350145578384, -0.01559256948530674, -0.03048374317586422, -0.023361878469586372, -0.014999081380665302, -0.021284667775034904, 0.035339560359716415, -0.02529071643948555, -0.008335820399224758, 0.027233043685555458, 0.009873495437204838, -0.011539311148226261, -0.0010369194205850363, 0.045024219900369644, 0.025614438578486443, 0.007276982069015503, -0.023105598986148834, -0.01532280258834362, 0.017386524006724358, -0.003699188819155097, -0.011202101595699787, 0.008726983331143856, -0.019220944494009018, -0.010547914542257786, 0.015902802348136902, -0.002776920795440674, 0.03242607042193413, 0.021891644224524498, -0.0008691576076671481, -0.010514194145798683, 0.005746050737798214, -0.023982344195246696, 0.009893728420138359, 0.004761398769915104, 0.022134434431791306, 0.014715825207531452, -0.02473769336938858, 0.0036486072931438684, -0.010453496128320694, 0.017103267833590508, 0.021945597603917122, 0.024494903162121773, -0.00439046835526824, -0.0035710493102669716, -0.015268849208950996, 0.019072571769356728, -0.014985593035817146, 0.022147923707962036, 0.005112096667289734, 0.0025611065793782473, -0.01866792142391205, -0.013023032806813717, 0.044053055346012115, -0.037713516503572464, -0.032587930560112, 0.0015907860361039639, 0.022458156570792198, -0.03879258781671524, -0.005071631632745266, -0.030456766486167908, 0.026612577959895134, 0.01004884485155344, 0.002357094781473279, -0.023806994780898094, 0.018195828422904015, 0.033235374838113785, 0.027017230167984962, -0.003685700474306941, 0.01116838026791811, 0.008436983451247215, 0.006245120894163847, 0.02092048153281212, -0.033370256423950195, -0.015268849208950996, -0.02288978546857834, 0.02135210856795311, -0.016671640798449516, 0.008531401865184307, 0.018209315836429596, -0.011181868612766266, -0.012105823494493961, 0.024521879851818085, -0.022822342813014984, -0.0023014552425593138, 0.0005770498537458479, -0.024953506886959076, -0.014526987448334694, -0.006609307136386633, -0.0043061659671366215, -0.017373036593198776, -0.020178619772195816, -0.027920952066779137, 0.00872023869305849, -0.009806053712964058, 0.013643498532474041, -0.009226053021848202, 0.03962886705994606, -0.025101879611611366, 0.05017678067088127, 0.03204839676618576, -0.011485357768833637, -0.001252733520232141, -0.002529071643948555, 0.030294906347990036, -0.014553964138031006, -0.048881895840168, -0.018640944734215736, 0.04313584789633751, 0.024076761677861214, 0.01996280625462532, 0.04216468334197998, 0.05322515591979027, -0.007897447794675827, 0.0033737816847860813, -0.03223723545670509, 0.017642803490161896, -0.0007713668746873736, -0.039143286645412445, -0.02248513326048851, -0.02529071643948555, -0.0030197114683687687, -0.011458381079137325, -0.0022491877898573875, -0.06118330359458923, -0.03272281587123871, -0.002883141627535224, -0.010311868041753769, -0.03962886705994606, 0.04262328892946243, 0.014783266931772232, -0.003375467611476779, 0.022390713915228844, -0.005924772005528212, 0.03498886525630951, -0.031697697937488556, -0.013697451911866665, -0.006551981903612614, -0.009387914091348648, -0.01383907999843359, -0.027448857203125954, -0.0005361632211133838, -0.03803723677992821, 0.026504671201109886, 0.004201631061732769, 0.021743271499872208, 0.007189307827502489, -0.002667327644303441, 0.0016379954759031534, -0.016820013523101807, -0.02516932226717472, -0.03005211614072323, -0.000710669148247689, -0.0038104678969830275, -0.0004906398826278746, -0.03269583731889725, 0.0418679378926754, -0.019760480150580406, 0.006265353411436081, 0.03250700235366821, -0.03358607366681099, -0.014041406102478504, 0.005948376376181841, -0.0418679378926754, -0.060481905937194824, -0.0015823558205738664, -0.056273531168699265, 0.0307804886251688, 0.015902802348136902, 0.0013083730591461062, 0.0022963969968259335, -0.0483153834939003, -0.013265823945403099, -0.013933499343693256, 0.010237682610750198, 0.010338844731450081, 0.023658622056245804, -0.04060003161430359, 0.04367538169026375, -0.056273531168699265, -0.01383907999843359, -0.019746990874409676, 0.020542806014418602, -0.006032678764313459, 0.04246142879128456, -0.0030163393821567297, -0.02011117711663246, 0.02274141274392605, -0.02066420204937458, 0.010972798801958561, 0.013474893756210804, -0.02274141274392605, -0.00600907439365983, -0.08772844076156616, 0.010912101715803146, 0.000677791191264987, 0.027502810582518578, 0.013623266480863094, 0.02317304164171219, 0.037173982709646225, 0.0026588973123580217, -0.01659071072936058, -0.0018428502371534705, 0.0024177925661206245, 0.020475365221500397, 0.03223723545670509, -0.013495126739144325, -0.009165355935692787, 0.018384665250778198, -0.0220669936388731, 0.013178149238228798, 0.007398377638310194, -0.005779771599918604, 0.0296474639326334, 0.002367211040109396, -0.03501584008336067, -0.00526384124532342, 0.03390979394316673, 0.012139543890953064, -0.007877214811742306, 0.016671640798449516, -0.0005538666737265885, -0.024966996163129807, -0.006238376721739769, 0.0296474639326334, 0.015417221002280712, 0.019787456840276718, 0.019288387149572372, 0.004373608157038689, 0.016536757349967957, -0.040977705270051956, -0.017750710248947144, 0.022377226501703262, -0.017656292766332626, 0.03736281767487526, 0.019301874563097954, -0.04211072996258736, -0.0035811655689030886, 0.03879258781671524, -0.01617257110774517, -0.01628047786653042, 0.013576056808233261, -0.005880934651941061, -0.032587930560112, 0.03221025690436363, 0.01586233824491501, 0.009920705109834671, 0.0029826185200363398, -0.019733503460884094, 0.007560238242149353, 0.009664425626397133, 0.0031191883608698845, 0.027017230167984962, -0.028352579101920128, 0.0039554680697619915, -0.02502094954252243, -0.015282337553799152, -0.023280948400497437, -0.004899655003100634, 0.0015191290294751525, -0.02051582932472229, -0.015633035451173782, 0.04893584921956062, 0.02078559622168541, -0.011289776302874088, -0.009940938092768192, -0.006046167574822903, 0.04189491644501686, 0.014675360172986984, 0.04030328616499901, -0.022458156570792198, 0.017319083213806152, -0.03949398174881935, 0.03331630304455757, -0.022242343053221703, -0.022242343053221703, -0.013340010307729244, 0.012416056357324123, -0.006039422936737537, -0.022673970088362694, 0.006093376781791449, -0.014999081380665302, -0.0024582576006650925, 0.009320472367107868, -0.00991396140307188, 0.02108234167098999, 0.0388195626437664, 0.00372279342263937, 0.0038037237245589495, -0.020461875945329666, -0.016469314694404602, -0.001745059504173696, 0.03919724002480507, 0.00044638116378337145, -0.025641415268182755, 0.008504425175487995, 0.011984427459537983, -0.00836279708892107, 0.0009492448880337179, -0.017103267833590508, -0.004376980010420084, -0.03347816318273544, 0.03048374317586422, -0.004248840268701315, -0.010372566059231758, 0.007270237896591425, 0.0005955963861197233, 0.010642333887517452, -0.002832560334354639, 0.058971207588911057, -0.030402813106775284, -0.018195828422904015, -0.012773497961461544, -0.017400013282895088, -0.02333490177989006, 0.05891725420951843, 0.05632748454809189, 0.007263493724167347, 0.03385584056377411, 0.01841164194047451, 0.0017172396183013916, -0.0027431997004896402, -0.001015000743791461, 0.021406061947345734, 0.010089309886097908, 0.017359547317028046, -0.034907933324575424, 0.017602339386940002, 0.0048524453304708, -0.00456244545057416, -0.0054729110561311245, -0.016752570867538452, 0.025182809680700302, -0.001553693087771535, 0.0179935023188591, 0.03685025870800018, 0.003844188991934061, -0.048396315425634384, 0.0001950546575244516, 0.024643274024128914, -0.06981586664915085, 0.0133534986525774, -0.02051582932472229, -0.0087134949862957, -0.018897224217653275, 0.01370419654995203, 0.006831865757703781, -0.015295825898647308, 0.0009391286293976009, -0.03693119063973427, -0.006636284291744232, -0.007722098845988512, -0.022552575916051865, -0.021433038637042046, 0.032938629388809204, -0.02038094587624073, 0.01936931721866131, -0.04340561479330063, 0.01714373379945755, -0.016968384385108948, 0.0032608164474368095, -0.019881876185536385, 0.02937769703567028, 0.001163373002782464, 0.0026201182045042515, 0.004036398604512215, -0.007701866328716278, 0.02024606242775917, 0.013825591653585434, 0.016213035210967064, 0.030861418694257736, 0.006933028344064951, 0.03229118883609772, 0.007142098620533943, -0.010952566750347614, 0.006123725790530443, 0.012287916615605354, -0.017076291143894196, -0.03569025918841362, -0.0007713668746873736, 0.005398725159466267, -0.02078559622168541, 0.02951258048415184, 0.0022137807682156563, 0.025627925992012024, 0.028352579101920128, -0.008194192312657833, 0.02558746188879013, 0.024508390575647354, -0.01615908183157444, -0.008376285433769226, -0.020974434912204742, 0.004407329019159079, 0.02682839147746563, -0.028028858825564384, 0.007870471104979515, 0.01559256948530674, 0.02400932088494301, -0.04030328616499901, 0.0008666285430081189, -0.02038094587624073, 0.025236763060092926, -0.00779628474265337, 0.006649772636592388, -0.03444932773709297, -0.03868468105792999, -0.06301771849393845, -0.006946516688913107, -0.037039097398519516, 0.010190472938120365, 0.007148842792958021, 0.014149312861263752, 0.060751672834157944, 0.0027550021186470985, 0.015471174381673336, 0.016793036833405495, -0.016226524487137794, 0.008605588227510452, 0.02136559784412384, 0.05729864910244942, 0.008625820279121399, -0.00279040914028883, 0.015120476484298706, -0.012389078736305237, 0.022970715537667274, 0.0272735096514225, -0.0027195950970053673, 0.006130469962954521, -0.04003351926803589, 0.020461875945329666, 0.013724428601562977, 0.016496291384100914, -0.0022441295441240072, 0.027866998687386513, -0.017534896731376648, -0.05692097172141075, -0.022377226501703262, 0.0048052361235022545, 0.006312563084065914, -0.05422329530119896, 0.0023992459755390882, -0.007202796172350645, -0.012604893185198307, -0.012598149478435516, -0.016415361315011978, -0.05972655862569809, 0.012038380838930607, -0.0033400605898350477, 0.013960476033389568, 0.020960945636034012, -0.03946700692176819, 0.011080706492066383, -0.00935419276356697, 0.023577691987156868, 0.01418977789580822, 0.05381864681839943, 0.004343259148299694, -0.012247451581060886, 0.007998610846698284, 0.016361407935619354, 0.010905357077717781, 0.022930249571800232, -0.01412233617156744, -0.00709488894790411, -0.005847213789820671, 0.0430009625852108, -0.012422800064086914, 0.01897815428674221, 0.02570885606110096, 0.052334923297166824, -0.009327216073870659, -0.018870247527956963, 0.016118617728352547, 0.020326992496848106, 0.009313727729022503, -0.0028055834118276834, -0.032399095594882965, 0.0046231430023908615, 0.011546054854989052, 0.03051071986556053, 0.011687682941555977, -0.006423842161893845, 0.011141403578221798, 0.02937769703567028, 0.027893975377082825, -0.017076291143894196, 0.003665467957034707, -0.003709305077791214, 0.037983283400535583, 0.018897224217653275, -0.014715825207531452, -0.02135210856795311, -0.037983283400535583, 0.025088390335440636, -0.009239542298018932, -0.00998140312731266, 0.007243261206895113, -0.04464654624462128, 0.006578958593308926, 0.0663088858127594, -0.0388195626437664, -0.016523268073797226, 0.019140014424920082, 0.03852282091975212, 0.03164374455809593, 0.023375365883111954, -0.016347918659448624, -0.01377163827419281, -0.0034091887064278126, -0.004060002975165844, 0.024130715057253838, 0.012260939925909042, 0.017588850110769272, 0.010311868041753769, -0.030106069520115852, -0.00794465746730566, 0.022957226261496544, 0.027893975377082825, -0.0005774713936261833, -0.03364002704620361, 0.028298625722527504, -0.0024768041912466288, -0.009637448936700821, 0.01208559051156044, 0.02346978522837162, -0.021325131878256798, 0.029836300760507584, -0.027610719203948975, 0.004029654432088137, -0.006919539999216795, -0.022862808778882027, 0.009610472247004509, 0.0014297685120254755, 0.027367927134037018, -0.02036745660007, 0.020691178739070892, 0.06458237767219543, 0.007951401174068451, 0.01967955008149147, 0.0024211646523326635, 0.007303959224373102, 0.008436983451247215, 0.01884327083826065, -0.022390713915228844, -0.046427011489868164, -0.044484686106443405, -0.01138419471681118, 0.014864197000861168, -0.004606282338500023, 0.019490713253617287, -0.01046024076640606, -0.005166050512343645, -0.033397234976291656, -0.00801209919154644, 0.028568394482135773, -0.02345629595220089, -0.00036671539419330657, -0.013340010307729244, -0.020758619531989098, -0.02669350802898407, -0.04211072996258736, 0.011492101475596428, 0.003358607180416584, -0.0019423270132392645, -0.013232102617621422, 0.0066464003175497055, -0.020960945636034012, 0.011525822803378105, 0.0049805850721895695, -0.0505814328789711, 0.018357688561081886, 0.004943491891026497, 0.012409311719238758, 0.0379023551940918, -0.006727330852299929, 0.007546749897301197, -0.008983262814581394, 0.008403262123465538, -0.03185955807566643, 0.010507449507713318, 0.04421491548418999, -0.010123031213879585, 0.04213770478963852, 0.014769778586924076, 0.0009888670174404979, 0.1013786792755127, 0.0062889582477509975, -0.020475365221500397, -0.02742188051342964, 0.01785861887037754, 0.010123031213879585, 0.04683166369795799, 0.013306288979947567, 0.03285769745707512, -0.013690708205103874, -0.01272628828883171, 0.0002600728766992688, -0.01980094611644745, 0.01783164218068123, -0.0065047722309827805, 0.0011490415781736374, -0.016334431245923042, -0.016671640798449516, -0.013528847135603428, -0.025371646508574486, -0.026612577959895134, -0.0032641885336488485, -0.009239542298018932, 0.007121865637600422, -0.004296049941331148, 0.0027533159591257572, -0.02697676420211792, 0.021176761016249657, 0.0030466883908957243, 0.017453966662287712, 0.022957226261496544, 0.010979543440043926, -0.01671210490167141, 0.035771191120147705, -0.03334328159689903, 0.016833500936627388, -0.019504200667142868, 0.010561402887105942, -0.02233676053583622, 0.02597862482070923, -0.05257771536707878, -0.02558746188879013, 0.029755370691418648, -0.001163373002782464, 0.03482700139284134, -0.0153902443125844, -0.0006520790047943592, 0.016415361315011978, 0.011694427579641342, 0.0215004812926054, -0.026464205235242844, 0.02163536474108696, 0.000814782571978867, 0.015363267622888088, -0.052604690194129944, -0.007317447569221258, -0.04122049733996391, -0.02136559784412384, 0.022512109950184822, -0.018182339146733284, 0.010015123523771763, -0.015430709347128868, -0.0028966302052140236, 0.0061405859887599945, 0.041652124375104904, 0.002598199527710676, 0.012746521271765232, -0.011026752181351185, 0.03129304572939873, -0.020987922325730324, -0.028757231310009956, 0.03525863215327263, 0.02980932407081127, -0.019720014184713364, 0.04820747673511505, 0.01911303773522377, 0.023793505504727364, -0.015012569725513458, -0.010993031784892082, -0.07564284652471542, 0.00020706774375867099, 0.016766058281064034, 0.03048374317586422, -7.334307883866131e-05, -0.013387219049036503, -0.011370706371963024, 0.0010335473343729973, -0.021864667534828186, -0.041085612028837204, -0.014108847826719284, -0.0008657855214551091, 0.03455723449587822, -0.025897694751620293, 0.012894893065094948, -0.006420469842851162, -0.013866056688129902, 0.008322332054376602, 0.02417118102312088, 0.026814904063940048, 0.005948376376181841, -0.0011220647720620036, -0.002171629574149847, 0.00956326350569725, -0.026720484718680382, -0.0027347696013748646, -0.04286607727408409, 0.010338844731450081, 0.013515358790755272, -0.000260283617535606, -0.03431444615125656, 0.03852282091975212, -0.02415769174695015, 0.004161166027188301, 0.0004170860629528761, 0.031967464834451675, -0.017872106283903122, 0.013434428721666336, -0.013097219169139862, -0.006181051023304462, -0.012861172668635845, 0.013724428601562977, -0.012362102046608925, -0.018209315836429596, 1.4654126516688848e-06, -0.010062333196401596, 0.058809347450733185, -0.011020008474588394, 0.02418467029929161, 0.032533977180719376, -0.03987165912985802, -0.0064643071964383125, -0.05125585198402405, -0.03458421304821968, -0.04974515363574028, -0.00604279525578022, 0.007391633465886116, 0.016793036833405495, -0.014149312861263752, -0.030240952968597412, 0.02035396918654442, -0.007674889639019966, -0.000922268140129745, -0.011060473509132862, 0.025074902921915054, 0.00734442425891757, 0.034206535667181015, 0.017534896731376648, -0.004191514570266008, -0.009745356626808643, -0.02937769703567028, -0.016401872038841248, -0.02429257705807686, -0.006191167514771223, -0.01588931493461132, 0.022228853777050972, -0.0053177946247160435, -0.03523165360093117, -0.025614438578486443, -0.01996280625462532, -0.024508390575647354, 0.010514194145798683, 0.0005332126165740192, 0.022512109950184822, -0.009617216885089874, -0.007459075190126896, -0.012065357528626919, -0.03167072311043739, 0.01559256948530674, -0.02400932088494301, -0.027340950444340706, 0.010500705800950527, -0.030267929658293724, -0.018195828422904015, 0.022579552605748177, 0.013454661704599857, -0.01017698459327221, 0.004876050166785717, 0.02937769703567028, 0.04084281995892525, 0.029728394001722336, 0.012685823254287243, -0.02924281358718872, 0.0037531424313783646, 0.015403732657432556, -0.007054423913359642, 0.008625820279121399, -0.04089677333831787, 0.0006263667601160705, -0.031535837799310684, -0.007486052345484495, 0.04229956865310669, -0.008551633916795254, 0.03509677201509476, -0.013029777444899082, -0.04232654348015785, -0.000763779622502625, 0.020192109048366547, -0.005543725099414587, -0.02487257681787014, -0.022444667294621468, -0.0384688675403595, -0.005068259779363871, 0.00801884289830923, 0.0025206415448337793, -0.013103963807225227, 0.014810243621468544, 0.012470009736716747, 0.010001635178923607, 0.005358259659260511, -0.006902679800987244, 0.007155586965382099, 0.013960476033389568, 0.018290245905518532, 0.04901678115129471, -0.008329075761139393, 0.02569536864757538, -0.009421635419130325, 0.005193027202039957, -0.018991641700267792, 0.014014429412782192, 0.03887351602315903, -0.013016289100050926, 0.02346978522837162, -0.025196298956871033, -0.04008747264742851, -0.003200118662789464, -0.027570253238081932, -0.02318652905523777, 0.03641863167285919, -0.01447303406894207, 0.026814904063940048, -0.0006866429466754198, 0.013272568583488464, -0.05014980584383011, 0.0071286098100245, 0.028028858825564384, -0.008956286124885082, -0.027732113376259804, -0.012523963116109371, 0.037443749606609344, -0.0032540722750127316, -0.02148699387907982, 0.000961890269536525, 0.01586233824491501, 0.019153503701090813, 0.013623266480863094, 0.030672581866383553, -0.018047455698251724, 0.006649772636592388, 0.009576751850545406, 0.01853303797543049, -0.021567923948168755, 0.008349308744072914, 0.023793505504727364, 0.01713024638593197, 0.014014429412782192, -0.012416056357324123, -0.022862808778882027, 0.007627679966390133, -0.005631399806588888, -0.01574094220995903, -0.005877562798559666, -0.0018327339785173535, 0.026760950684547424, -0.0024110483936965466, 0.006400237325578928, 0.00642721401527524, -0.021244201809167862, 0.012193497270345688, -0.043081894516944885, -0.024346530437469482, -0.02541211247444153, -0.033505141735076904, -0.005884306970983744, 0.038253054022789, 0.002284594811499119, 0.005395352840423584, -0.0013808731455355883, 0.003668840043246746, 0.0032810489647090435, 0.01994931697845459, 0.0250074602663517, 0.01440559234470129, 0.01370419654995203, -0.017885595560073853, -0.023577691987156868, 0.014567452482879162, 0.03223723545670509, -0.03496188670396805, 0.01757536269724369, -0.012166520580649376, 0.023119086399674416, -0.020731642842292786, -0.00779628474265337, 0.01068279892206192, -0.002618432277813554, -0.006423842161893845, -0.05746050924062729, 0.01953117735683918, -0.006245120894163847, 0.023523738607764244, -0.014877685345709324, 0.011283031664788723, 0.010662565939128399, 0.0022424436174333096, 0.0191265270113945, -0.009596983902156353, -0.02163536474108696, 0.020596759393811226, -0.011876520700752735, 0.007189307827502489, -0.003028141800314188, -0.03666142374277115, 0.043621428310871124, 0.025398623198270798, 0.0424344502389431, -0.007607447449117899, -0.003434479236602783, -0.056705158203840256, 0.023159552365541458, -0.01798001304268837, 0.021567923948168755, -0.006865586619824171, -0.021244201809167862, -0.0483153834939003, 0.001438198727555573, 0.006184423342347145, -0.01096605509519577, 0.011431404389441013, 0.004801864270120859, 0.002185117918998003, -0.007175819482654333, 0.03763258829712868, 0.006582330446690321, 0.014081871137022972, -0.01575442962348461, -0.027462346479296684, -0.02499397285282612, -0.007843494415283203, 0.022566063329577446, 0.009799310006201267, -0.007034191396087408, 0.010696287266910076, 0.03064560517668724, 0.0005075003718957305, -0.0007334307883866131, -0.00353058404289186, -0.00664302846416831, -0.000436264876043424, 0.00656547024846077, -0.01868140883743763, 0.021999550983309746, -0.003992561250925064, 0.027354439720511436, 0.009050704538822174, -0.004012793768197298, 0.016941407695412636, -0.019504200667142868, -0.048558175563812256, 0.011822567321360111, 0.036310724914073944, -0.027367927134037018, 0.02864932455122471, 0.005567329935729504, -0.012402567081153393, 0.010163496248424053, -0.01377163827419281, -0.024090250954031944, 0.008187447674572468, 0.011957450769841671, -0.030321883037686348, -0.009644193574786186, -0.007964889518916607, -0.033370256423950195, -0.024225134402513504, 0.013758149929344654, 0.02655862458050251, 0.006885819137096405, -0.045024219900369644, 0.005820237100124359, 0.013090475462377071, -0.0017602338921278715, 0.010932333767414093, -0.01011628657579422, 0.016401872038841248, -0.008477448485791683, -0.0028645952697843313, -0.014203266240656376, -0.05773027613759041]
match_threshold = 1.0
match_count = 5

# Define the SQL query using placeholders (%s)
query2 = sql.SQL("""
                 select *
from information_schema.routines
where routine_name='match_documents'
                 """)

query = sql.SQL("""
    SELECT *
    FROM match_documents(
        %s::vector(1536),  -- pass the query embedding
        %s,  -- match threshold
        %s   -- match count
    )
""")

# Use context managers to handle database connection and cursor
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        # Execute the query with the provided parameters
        cur.execute(query, (query_embedding, match_threshold, match_count))
        
        # Fetch all results
        results = cur.fetchall()

# Process the results
for row in results:
    print(row)