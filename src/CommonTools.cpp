#include "../include/CommonTools.h"
#include <iostream>
#include <deque>
#include <queue>
#include <Eigen/SPQRSupport>
#include <igl/per_vertex_normals.h>
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/boundary_loop.h>
#include <filesystem>

void quadS3(double w, std::vector<QuadraturePoints>& quadLists)
{
	QuadraturePoints point;
	point.u = 1. / 3.;
	point.v = 1. / 3.;
	point.weight = w;
}

void quadS21(double x, double w, std::vector<QuadraturePoints>& quadLists)
{
	QuadraturePoints point;

	double pos[3];
	point.weight = w;
	pos[0] = x;
	pos[1] = x;
	pos[2] = 1 - 2 * x;
	for (int i = 0; i < 3; i++)
	{
		point.u = pos[i];
		point.v = pos[(i + 1) % 3];
		quadLists.push_back(point);
	}
}

void quadS111(double x, double y, double w, std::vector<QuadraturePoints>& quadLists)
{
	QuadraturePoints point;
	double pos[3];
	point.weight = w;
	pos[0] = x;
	pos[1] = y;
	pos[2] = 1 - x - y;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 1; j < 3; j++)
		{
			point.u = pos[i];
			point.v = pos[(i + j) % 3];
			quadLists.push_back(point);
		}
	}
}

std::vector<QuadraturePoints> buildQuadraturePoints(int order)
{
	std::vector<QuadraturePoints> quadLists;
	quadLists.clear();
	 if(order == 1)                // num of points = 1
	 {
		 quadLists.clear();
		 quadS3(1., quadLists);
	 }
	 else if(order == 2)           // num of points = 3
	 {
		 quadLists.clear();
		 quadS21(1.0/ 6, 1.0 / 3, quadLists);
	 }
	 else if(order == 3)           // num of points = 6
	 {
		 quadLists.clear();
		 quadS21(.16288285039589191090016180418490635, .28114980244097964825351432270207695, quadLists);
		 quadS21(.47791988356756370000000000000000000, .05218353089235368507981901063125638, quadLists);
	 }
	 else if(order == 4)
	 {
		 quadLists.clear();
		 quadS21(.44594849091596488631832925388305199, .22338158967801146569500700843312280, quadLists);
		 quadS21(.09157621350977074345957146340220151, .10995174365532186763832632490021053, quadLists);
	 }
	 else if(order == 5)         // num of points = 7
	 {
		 quadLists.clear();
		 quadS21(.10128650732345633880098736191512383, .12593918054482715259568394550018133, quadLists);
		 quadS21(.47014206410511508977044120951344760, .13239415278850618073764938783315200, quadLists);
		 quadS3(9./40., quadLists);
	 }
	 else if(order == 6)        // num of points = 12
	 {
		 quadLists.clear();
		 quadS21(.06308901449150222834033160287081916, .05084490637020681692093680910686898, quadLists);
		 quadS21(.24928674517091042129163855310701908, .11678627572637936602528961138557944, quadLists);
		 quadS111(.05314504984481694735324967163139815, .31035245103378440541660773395655215, .08285107561837357519355345642044245, quadLists);
	 }
	 else if(order == 7)         // num of points = 15
	 {
		 quadS21(.02826392415607634022359600691324002, .01353386251566556156682309245259393, quadLists);
		 quadS21(.47431132326722257527522522793181654, .07895125443201098137652145029770332, quadLists);
		 quadS21(.24114332584984881025414351267036207, .12860792781890607455665553308952344, quadLists);
		 quadS111(.76122274802452380000000000000000000, .04627087779880891064092559391702049, .05612014428337535791666662874675632, quadLists);
	 }
	 else if(order == 8)         // num of points = 16
	 {
		 quadS3(.14431560767778716825109111048906462, quadLists);
		 quadS21(.17056930775176020662229350149146450, .10321737053471825028179155029212903, quadLists);
		 quadS21(.05054722831703097545842355059659895, .03245849762319808031092592834178060, quadLists);
		 quadS21(.45929258829272315602881551449416932, .09509163426728462479389610438858432, quadLists);
		 quadS111(.26311282963463811342178578628464359, .00839477740995760533721383453929445, .02723031417443499426484469007390892, quadLists);
	 }
	 else if (order == 9)        // num of points = 19
	 {
		 quadS3(.09713579628279883381924198250728863, quadLists);
		 quadS21(.48968251919873762778370692483619280, .03133470022713907053685483128720932, quadLists);
		 quadS21(.04472951339445270986510658996627636, .02557767565869803126167879855899982, quadLists);
		 quadS21(.43708959149293663726993036443535497, .07782754100477427931673935629940396, quadLists);
		 quadS21(.18820353561903273024096128046733557, .07964773892721025303289177426404527, quadLists);
		 quadS111(.74119859878449802069007987352342383, .22196298916076569567510252769319107, .04328353937728937728937728937728938, quadLists);
	 }
	 else if (order == 10)       // num of points = 25
	 {
		 quadS3(.08093742879762288025711312381650193, quadLists);
		 quadS21(.42727317884677553809044271751544715, .07729858800296312168250698238034344, quadLists);
		 quadS21(.18309922244867502052157438485022004, .07845763861237173136809392083439673, quadLists);
		 quadS21(.49043401970113058745397122237684843, .01746916799592948691760716329067815, quadLists);
		 quadS21(.01257244555158053273132908502104126, .00429237418483282803048040209013191, quadLists);
		 quadS111(.65426866792006614066657009558762790, .30804600168524770000000000000000000, .03746885821046764297902076548504452, quadLists);
		 quadS111(.12280457706855927343012981748128116, .03337183373930478624081644177478038, .02694935259187995964544947958109671, quadLists);
	 }
	 else if (order == 11)      // num of points = 28
	 {
		 quadS3(.08117796029686715951547596874982357, quadLists);
		 quadS21(.03093835524543078489519501499130475, .01232404350690949411847390101623284, quadLists);
		 quadS21(.43649818113412884191761527655997324, .06282800974441010728333942816029398, quadLists);
		 quadS21(.49898476370259326628798698383139087, .01222037904936452975521221500393789, quadLists);
		 quadS21(.21468819795859433660687581387825086, .06770134895281150992098886182322559, quadLists);
		 quadS21(.11368310404211339020529315622836178, .04021969362885169042356688960756866, quadLists);
		 quadS111(.82561876616486290435880620030835800, .15974230459185018980086078822500751, .01476227271771610133629306558778206, quadLists);
		 quadS111(.64047231013486526767703659081896681, .31178371570959900000000000000000000, .04072799645829903966033695848161786, quadLists);
	 }
	 else if (order == 12)       // num of points = 33
	 {
		 quadS21(.02131735045321037024685697551572825, .00616626105155901723386648378523035, quadLists);
		 quadS21(.27121038501211592234595134039689474, .06285822421788510035427051309288255, quadLists);
		 quadS21(.12757614554158592467389632515428357, .03479611293070894298932839729499937, quadLists);
		 quadS21(.43972439229446027297973662348436108, .04369254453803840213545726255747497, quadLists);
		 quadS21(.48821738977380488256466206525881104, .02573106644045533541779092307156443, quadLists);
		 quadS111(.69583608678780342214163552323607254, .28132558098993954824813069297455275, .02235677320230344571183907670231999, quadLists);
		 quadS111(.85801403354407263059053661662617818, .11625191590759714124135414784260182, .01731623110865889237164210081103407, quadLists);
		 quadS111(.60894323577978780685619243776371007, .27571326968551419397479634607976398, .04037155776638092951782869925223677, quadLists);
	 }
	 else if (order == 13)      // num of points = 37
	 {
		 quadS3(.06796003658683164428177442468088488, quadLists);
		 quadS21(.42694141425980040602081253503137421, .05560196753045332870725746601046147, quadLists);
		 quadS21(.22137228629183290065481255470507908, .05827848511919998140476708351333981, quadLists);
		 quadS21(.02150968110884318386929131353405208, .00605233710353917184179280003229082, quadLists);
		 quadS21(.48907694645253934990068971909020439, .02399440192889473077371079945095965, quadLists);
		 quadS111(.62354599555367557081585435318623659, .30844176089211777465847185254124531, .03464127614084837046598682851091822, quadLists);
		 quadS111(.86470777029544277530254595089569318, .11092204280346339541286954522167452, .01496540110516566726324585713290344, quadLists);
		 quadS111(.74850711589995219517301859578870965, .16359740106785048023388790171095725, .02417903981159381913744574557306076, quadLists);
		 quadS111(.72235779312418796526062013230478405, .27251581777342966618005046435408685, .00959068100354326272259509016611089, quadLists);
	 }
	 else if (order == 14)       // num of points = 42
	 {
		 quadS21(.17720553241254343695661069046505908, .04216258873699301753823043732418613, quadLists);
		 quadS21(.01939096124870104817825009505452951, .00492340360240008168182602350904215, quadLists);
		 quadS21(.06179988309087260126747882843693579, .01443369966977666760170992148065332, quadLists);
		 quadS21(.41764471934045392250944082218564344, .03278835354412535064131097873862534, quadLists);
		 quadS21(.48896391036217863867737602045239024, .02188358136942889064084494596332597, quadLists);
		 quadS21(.27347752830883865975494428326269856, .05177410450729158631478491016639640, quadLists);
		 quadS111(.17226668782135557837528960161365733, .05712475740364793903567712421891471, .02466575321256367396287524518363623, quadLists);
		 quadS111(.57022229084668317349769621336235426, .09291624935697182475824858954872035, .03857151078706068322848902781041086, quadLists);
		 quadS111(.29837288213625775297083151805961273, .01464695005565440967054132792007421, .01443630811353384049608869199901580, quadLists);
		 quadS111(.11897449769695684539818196192990548, .00126833093287202508724640109549269, .00501022883850067176986009308248912, quadLists);
	 }
	 else if (order == 15)    // num of points = 49
	 {
		 quadS3(.02357126703190634206659321140821418, quadLists);
		 quadS21(.11022229622834687297855264132259850, .01517314955721170450311858877690239, quadLists);
		 quadS21(.05197643301003435047003197947889073, .01297600128392884154979521077280757, quadLists);
		 quadS21(.49114565807532554119014945122395425, .01706629596800615670942600046160914, quadLists);
		 quadS21(.39315718888435884048226809785071794,.04576001946273760698482638108892258, quadLists);
		 quadS111(.03737440487572919066543605209836625,.96251835223001214880811969560396873, .00222757447282223154006065426298478, quadLists);
		 quadS111(.24824877798467321198263980694374938,.19316669854521416819773100288721521,.02701014165986947101315702212247500, quadLists);
		 quadS111(.20699402274830217740486528153682148,.08689590883549962551575259619781217,.02608377963958756403057720483642768, quadLists);
		 quadS111(.14854110526954708137688902238435510,.01743682539845430796259020511767948, .01211015327702828337230795926322736, quadLists);
		 quadS111(.30674237923596382376588728350286621, .01749251095825766163254977051260599, .01564785059680444573399007149035058, quadLists);
		 quadS111(.36703198754220473278855469116984882,.09034802175864556044634095119222305, .03417088937929479242522512890637806, quadLists);
	 }
	 else if (order == 16)       // num of points = 55
	 {
		 quadS3(.04802218868037709055183940458051988, quadLists),
		 quadS21(.08179498313137387264146559311886101, .01470910030680192710340364286186919, quadLists);
		 quadS21(.16530060196977965062676193293355656, .02954458654931925599530972679646409, quadLists);
		 quadS21(.46859210534946138669460289729660561, .02612501735108837749859756549171557, quadLists);
		 quadS21(.01443881344541668261410895669566020, .00278038735239000697500301613866207, quadLists);
		 quadS21(.24178428539178335340689445929320769, .03182177300053664950342729005594961, quadLists);
		 quadS21(.49531034298776996406549508687740551, .00864583434950965990117373416984893, quadLists);
		 quadS111(.65051340266135229943114468484168666, .33139974453708955658132316818259388, .01430033290449536514661642536825213, quadLists);
		 quadS111(.60401128149599703984940410303596702, .30324716274994218504155217807834692, .02784977720360082995222987342395349, quadLists);
		 quadS111(.80216825757474166361686194781166705, .18802805952123717344418211429398875, .00704167340663609756237018808928069, quadLists);
		 quadS111(.75650560644282839655115407575806082, .18350466852229686368238027743700035, .01789983825993372860177020907581078, quadLists);
		 quadS111(.46593843871411818488381073359154639, .35964594879750460000000000000001000, .02745820038434976307247003810091720, quadLists);
		 quadS111(.90639484399204150136249966186534000, .07719437129575543228251522505271386, .00729979693943176208411254408777766, quadLists);
	 }
	 else if (order == 17)      // num of points = 60
		 {
		 quadS21(.24056306963626902977934166278860247, .03829254008003568749425168889491817 , quadLists);
		 quadS21(.08092323589766073062004798772340524, .01669528699775339594318472807122019 , quadLists);
		 quadS21(.01001414912499135088254841140047604, .00143512454359061224492929722268097 , quadLists);
		 quadS21(.15437652078663289107430782196727737, .02864276849185053630399044294140648 , quadLists);
		 quadS21(.41716986201996268598941663596983268, .03408569078206214964786810427776196 , quadLists);
		 quadS21(.47086974573840098186867398532866671, .02467274200053089056925349793140004 , quadLists);
		 quadS21(.49811803384542204444865152799034832, .00586679757537134154263246190805349 , quadLists);
		 quadS21(.36473840565291924199871629076775930, .02321859500422896151112767944153052 , quadLists);
		 quadS111(.10986590708262616153720966373050601,.30466576969866569523225839525499357, .03084965458251406099116307348593810 , quadLists);
		 quadS111(.20493227462918790108024139159058423,.05248758390645425414786013344982922, .01881398544005420038782109445200127 , quadLists);
		 quadS111(.05813921564266244000000000000000000,.01500053995225954378593128753997425, .00512343450397285555007197439694996 , quadLists);
		 quadS111(.13859554086776482539309659376771751,.01501023347973182500884052064335399, .00701239348475201777118052342883162 , quadLists);
		 quadS111(.34660546952009260087829868774027952,.02336212893314653752768977049783837, .01538229443504461311363086994295179 , quadLists);
		 quadS111(.24821986889585591697209834974065293,.00000099999999999965762180770907324, .00303013148261713122418018061550803, quadLists);
		 }
	 else if (order == 18)       // num of points = 67
		 {
		 quadS3(.03074852123911585539935333820159969, quadLists);
		 quadS21(.15163850697260486492387353795772074, .02031833884545839730521676856098738 , quadLists);
		 quadS21(.07243870556733287047426206374480081, .01379028660476693880147269080330003 , quadLists);
		 quadS21(.00375894434106834585702462733286887, .00053200561694778056109294261721746 , quadLists);
		 quadS21(.41106710187591949855469549486746318, .03347199405984789811876973462144190 , quadLists);
		 quadS21(.26561460990537421478430796115175039, .03111639660200613119689389250158563 , quadLists);
		 quadS21(.47491821132404573588789755091754023, .01310702749173875567860153100348528 , quadLists);
		 quadS111(.06612245802840338770053947185398348,
				  .17847912556588763355267204638676643, .01691165391748007879456553323826843 , quadLists);
		 quadS111(.26857330639601384733212028806856623,
				  .14906691012577383920019113944789784, .02759288648857947802009593334620683 , quadLists);
		 quadS111(.30206195771287080772484323648551723,
				  .05401173533902423468044436247084948, .01636590841398656595815221611374510 , quadLists);
		 quadS111(.13277883027138932992144407050471004,
				  .01433152477894195356844867129563809, .00764170497271963595084711372125680 , quadLists);
		 quadS111(.25650615977424154068897765977748937,
				  .01050501881924193559868603344210775, .00772983528000622700809279634102600 , quadLists);
		 quadS111(.41106566867461836291309677848250996,
				  .01169182467466708527042342649785763, .00958612447436150376044024017260990 , quadLists);
		 quadS111(.04727614183265178252228403898505622,
				  .01249893248349544012804819357953175, .00421751677474444290984387716007124 , quadLists);
		 quadS111(.38504403441316367334400254247436861,
				  .52452892523249571422861434426430408, .01532825819455314086704628681920691, quadLists);

		 }
	 else if (order == 19)      // num of points = 73
		 {
		 quadS3(.03290633138891865208361434484647497, quadLists);
		 quadS21(.48960998707300633196613106574829817, .01033073189127205336703996357174833 , quadLists);
		 quadS21(.45453689269789266204675939053572830, .02238724726301639252918455603516271 , quadLists);
		 quadS21(.40141668064943118739399562381068860, .03026612586946807086528019098259122 , quadLists);
		 quadS21(.25555165440309761132218176810926787, .03049096780219778100003158657852042 , quadLists);
		 quadS21(.17707794215212955164267520651590115, .02415921274164090491184803098664001 , quadLists);
		 quadS21(.11006105322795186130008495167737397, .01605080358680087529162277027642948 , quadLists);
		 quadS21(.05552862425183967124867841247135571, .00808458026178406048180567324219442 , quadLists);
		 quadS21(.01262186377722866849023476677870599, .00207936202748478075134750167439841 , quadLists);
		 quadS111(.60063379479464500000000000000000000,
				  .39575478735694286230479469406582787, .00388487690498138975670499199277266 , quadLists);
		 quadS111(.13446675453077978561204319893264695,
				  .55760326158878396836395324250118097, .02557416061202190389292970195260027 , quadLists);
		 quadS111(.72098702581736505521665290233827892,
				  .26456694840652020804030173490121494, .00888090357333805774552592470351753 , quadLists);
		 quadS111(.59452706895587092461388928802650670,
				  .35853935220595058842492699064590088, .01612454676173139121978526932783766 , quadLists);
		 quadS111(.83933147368083857861749007714840520,
				  .15780740596859474473767360335950651, .00249194181749067544058464757594956 , quadLists);
		 quadS111(.22386142409791569130336938950653642,
				  .70108797892617336732328833655951158, .01824284011895057837766571320973615 , quadLists);
		 quadS111(.82293132406985663162747155916053316,
				  .14242160111338343731557475687723745, .01025856373619852130804807004235813 , quadLists);
		 quadS111(.92434425262078402945585913790156314,
				  .06549462808293770339232652498592557, .00379992885530191397907315371363970, quadLists);

		 }
	 else if (order == 20)       // num of points = 82
		 {
		 quadS3(.02343898837621685337578235989880370, quadLists);
		 quadS21(.47253969049374944291236394079678822, .01701187887065179140821050028462978 , quadLists);
		 quadS21(.43559179765819053474788158522088855, .02213462902539847149771471956849632 , quadLists);
		 quadS21(.38548317769095410374903697427852223, .02225012034148936704815054477635356 , quadLists);
		 quadS21(.18589787266938260089793207404361894, .02108801427518765728972259255778044 , quadLists);
		 quadS21(.10294309387227568202927226453268166, .01436673198237250674135242117943876 , quadLists);
		 quadS21(.04420435682210499536228665005842580, .00596064247309054870158068681368087 , quadLists);
		 quadS21(.01187700008194990884008379159291605, .00185446185638856226012710730597991 , quadLists);
		 quadS111(.42317710194393600041367501539791831,
				  .56624681421216737896202019229528315, .00766503799888173467975505965999393 , quadLists);
		 quadS111(.28717803493735130968896269265785627,
				  .70363134375008177073022062562769704, .00611096641377269564490800130795800 , quadLists);
		 quadS111(.16282486571070316869304103317873899,
				  .83237331116054735414989063179341648, .00329750122890750342409287108138500 , quadLists);
		 quadS111(.06468063449817511170284328732719589,
				  .92925769945837560706677619337951412, .00258504328745784243926241647780699 , quadLists);
		 quadS111(.33400627700511908113700959506140627,
				  .61323701639938793580192466071340766, .01604829966346833686116840419650012 , quadLists);
		 quadS111(.21092379552418127847921338445298798,
				  .75151051490601053625741794541954190, .00987379147789446687652271685071374 , quadLists);
		 quadS111(.11508038084136831598057023754719804,
				  .85015113156210283588089253042485358, .00817314741769008089505196969930370 , quadLists);
		 quadS111(.31075208646508429564535326048014931,
				  .56198974953953613108255692415961755, .02127212420812707803762062418100071 , quadLists);
		 quadS111(.20200619801045898902767636571938539,
				  .70171644023616188456192811677305342, .01629559968072678577307785994248523 , quadLists);
		 quadS111(.28902320790895929973608959414472710,
				  .49138856232319209839401293034098827, .01910541781474788066581373037653881, quadLists);
		 }
	 else
	 {
		 std::cerr << "We only provide quad points for the order from 1 to 20, input is: " << order << std::endl;
		 exit(1);
	 }
	 return quadLists;
}

Eigen::Vector3d computeHatWeight(double u, double v)
{
	Eigen::Vector3d weights;
	Eigen::Vector3d bary(1 - u - v, u, v);
	for (int i = 0; i < 3; i++)
	{
//		weights(i) = 3 * bary(i) * bary(i) - 2 * bary(i) * bary(i) * bary(i) + 2 * bary(i) * bary((i + 1) % 3) * bary((i + 2) % 3);
		            weights(i) = bary(i);
	}
	return weights;
}

Eigen::MatrixXd SPDProjection(Eigen::MatrixXd A)
{
	Eigen::MatrixXd posHess = A;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	es.compute(posHess);
	Eigen::VectorXd evals = es.eigenvalues();

	for (int i = 0; i < evals.size(); i++)
	{
		if (evals(i) < 0)
			evals(i) = 0;
	}
	Eigen::MatrixXd D = evals.asDiagonal();
	Eigen::MatrixXd V = es.eigenvectors();
	posHess = V * D * V.transpose();

	return posHess;
}

Eigen::VectorXd vertexVec2IntrinsicVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh)
{
	int nedges = mesh.nEdges();
	Eigen::VectorXd edgeOmega(nedges);

	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		Eigen::Vector3d e = pos.row(vid1) - pos.row(vid0);
		edgeOmega(i) = (v.row(vid0) + v.row(vid1)).dot(e) / 2;
	}
	return edgeOmega;
}

Eigen::MatrixXd vertexVec2IntrinsicHalfEdgeVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh)
{
	int nedges = mesh.nEdges();
	Eigen::MatrixXd edgeOmega(nedges, 2);

	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		Eigen::Vector3d e = pos.row(vid1) - pos.row(vid0);
		edgeOmega(i, 0) = v.row(vid0).dot(e);
		edgeOmega(i, 1) = -v.row(vid1).dot(e);
	}
	return edgeOmega;
}

struct UnionFind
{
	std::vector<int> parent;
	std::vector<int> sign;
	UnionFind(int items)
	{
		parent.resize(items);
		sign.resize(items);
		for (int i = 0; i < items; i++)
		{
			parent[i] = i;
			sign[i] = 1;
		}
	}

	std::pair<int, int> find(int i)
	{
		if (parent[i] != i)
		{
			auto newparent = find(parent[i]);
			sign[i] *= newparent.second;
			parent[i] = newparent.first;
		}

		return { parent[i], sign[i] };
	}

	void dounion(int i, int j, int usign)
	{
		auto xroot = find(i);
		auto yroot = find(j);
		if (xroot.first != yroot.first)
		{
			parent[xroot.first] = yroot.first;
			sign[xroot.first] = usign * xroot.second * yroot.second;
		}
	}
};

void combField(const Eigen::MatrixXi& F, const Eigen::MatrixXd& w, Eigen::MatrixXd& combedW)
{
	int nfaces = F.rows();
	int nverts = w.rows();

	UnionFind uf(nverts);
	MeshConnectivity mesh(F);

	struct Visit
	{
		int edge;
		int sign;
		double norm;
		bool operator<(const Visit& other) const
		{
			return norm > other.norm;
		}
	};

	std::priority_queue<Visit> pq;

	int nedges = mesh.nEdges();
	for (int i = 0; i < nedges; i++)
	{
		int vid1 = mesh.edgeVertex(i, 0);
		int vid2 = mesh.edgeVertex(i, 1);


		Eigen::Vector3d curw = w.row(vid1);
		Eigen::Vector3d nextw = w.row(vid2);

		double innerp = curw.dot(nextw);
		int sign = (innerp < 0 ? -1 : 1);

		double normcw = curw.norm();
		double normnw = nextw.norm();
		double negnorm = -std::min(normcw, normnw);
		pq.push({ i, sign, negnorm });


	}

	while (!pq.empty())
	{
		auto next = pq.top();
		pq.pop();
		uf.dounion(mesh.edgeVertex(next.edge, 0), mesh.edgeVertex(next.edge, 1), next.sign);
	}

	combedW.resize(nverts, 3);
	for (int i = 0; i < nverts; i++)
	{
		int sign = uf.find(i).second;
		combedW.row(i) = w.row(i) * sign;
	}

}

Eigen::MatrixXd intrinsicHalfEdgeVec2VertexVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh)
{
	int nedges = mesh.nEdges();
	int nverts = pos.rows();
	Eigen::MatrixXd vertOmega(nverts, 3);
	vertOmega.setZero();

    Eigen::MatrixXd vertNormals;
    igl::per_vertex_normals(pos, mesh.faces(), vertNormals);

	Eigen::SparseMatrix<double> A;
	std::vector<Eigen::Triplet<double>> T;

	Eigen::VectorXd edgeVec(2 * nedges);

	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		Eigen::Vector3d e = pos.row(vid1) - pos.row(vid0);
		edgeVec.segment<2>(2 * i) = v.row(i);
		for (int j = 0; j < 3; j++)
		{
			T.push_back({ 2 * i, 3 * vid0 + j, e(j) });
			T.push_back({ 2 * i + 1, 3 * vid1 + j, -e(j) });
		}
	}
	A.resize(2 * nedges, 3 * nverts);
	A.setFromTriplets(T.begin(), T.end());

    Eigen::SparseMatrix<double> AT, AAT;
    AT = A.transpose();
    AAT = AT * A;

    Eigen::VectorXd ATb = AT * edgeVec;

    T.clear();
    for (int k=0; k<AAT.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(AAT,k); it; ++it)
        {
            T.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
        }

    for(int i = 0; i < nverts; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            T.push_back({3 * i + j, 3 * nverts + i, vertNormals(i, j)});
            T.push_back({3 * nverts + i, 3 * i + j, vertNormals(i, j)});
        }
    }

    A.resize(4 * nverts, 4 * nverts);
    A.setFromTriplets(T.begin(), T.end());

    Eigen::VectorXd rhs(4 * nverts);
    rhs.setZero();
    rhs.segment(0, 3 * nverts) = ATb;

	//Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver(A);
	/*Eigen::SPQR<Eigen::SparseMatrix<double>> solver(A);
	Eigen::VectorXd sol = solver.solve(rhs);

	for (int i = 0; i < nverts; i++)
	{
		vertOmega.row(i) = sol.segment<3>(3 * i);
	}*/
	return vertOmega;
}

Eigen::MatrixXd intrinsicHalfEdgeVec2FaceVec(const Eigen::MatrixXd& w, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh)
{
    int nfaces = mesh.nFaces();

    Eigen::MatrixXd faceVec = Eigen::MatrixXd::Zero(nfaces, 3);
    for(int i = 0; i < nfaces; i++)
    {
//        std::cout << "face: " << i  << " of total faces " << nfaces << std::endl;
        for(int j = 0; j < 3; j++)
        {
            int vid = mesh.faceVertex(i, j);

            int eid0 = mesh.faceEdge(i, (j + 1) % 3);
            int eid1 = mesh.faceEdge(i, (j + 2) % 3);

            Eigen::Vector3d e0 = pos.row(mesh.faceVertex(i, (j + 2) % 3)) - pos.row(vid);
            Eigen::Vector3d e1 = pos.row(mesh.faceVertex(i, (j + 1) % 3)) - pos.row(vid);

            int flag0 = 0, flag1 = 0;
            Eigen::Vector2d rhs;

            if (mesh.edgeVertex(eid0, 0) == vid)
            {
                flag0 = 0;
                rhs(0) = w(eid0, 0);
            }
            else
            {
                flag0 = 1;
                rhs(0) = w(eid0, 1);
            }


            if (mesh.edgeVertex(eid1, 0) == vid)
            {
                flag1 = 0;
                rhs(1) = w(eid1, 0);
            }
            else
            {
                flag1 = 1;
                rhs(1) = w(eid1, 1);
            }

            Eigen::Matrix2d I;
            I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
            Eigen::Vector2d sol = I.inverse() * rhs;

            faceVec.row(i) += (sol(0) * e0 + sol(1) * e1) / 3;
        }
    }
    return faceVec;
}

Eigen::MatrixXd intrinsicEdgeVec2FaceVec(const Eigen::VectorXd& w, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh)
{
	int nfaces = mesh.nFaces();

	Eigen::MatrixXd faceVec = Eigen::MatrixXd::Zero(nfaces, 3);
	for (int i = 0; i < nfaces; i++)
	{
		//        std::cout << "face: " << i  << " of total faces " << nfaces << std::endl;
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);

			int eid0 = mesh.faceEdge(i, (j + 1) % 3);
			int eid1 = mesh.faceEdge(i, (j + 2) % 3);

			Eigen::Vector3d e0 = pos.row(mesh.faceVertex(i, (j + 2) % 3)) - pos.row(vid);
			Eigen::Vector3d e1 = pos.row(mesh.faceVertex(i, (j + 1) % 3)) - pos.row(vid);

			int flag0 = 1, flag1 = 1;
			Eigen::Vector2d rhs;

			if (mesh.edgeVertex(eid0, 0) == vid)
			{
				flag0 = 1;
			}
			else
			{
				flag0 = -1;
			}


			if (mesh.edgeVertex(eid1, 0) == vid)
			{
				flag1 = 1;
			}
			else
			{
				flag1 = -1;
			} 
			rhs(0) = flag0 * w(eid0);
			rhs(1) = flag1 * w(eid1);

			Eigen::Matrix2d I;
			I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
			Eigen::Vector2d sol = I.inverse() * rhs;

			faceVec.row(i) += (sol(0) * e0 + sol(1) * e1) / 3;
		}
	}
	return faceVec;
}

double unitMagEnergy(const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess, bool isProj)
{
	int nverts = zvals.size();
	double energy = 0;
	
	if (deriv)
		deriv->setZero(2 * nverts);

	if (hess)
		hess->clear();

	double sum = 0;

	Eigen::VectorXd derivSum(2 * nverts);

	for (int i = 0; i < nverts; i++)
	{
		sum += zvals[i].real()* zvals[i].real() + zvals[i].imag() * zvals[i].imag();
		if (deriv)
		{
			derivSum.segment<2>(2 * i) << 2 * zvals[i].real(), 2 * zvals[i].imag();
		}
	}

	energy = 0.5 * (sum - 1) * (sum - 1);

	if (deriv)
		*deriv = (sum - 1) * derivSum;

	if (hess)
	{
		// hess is dense. Really bad
	}

	return energy;
}

Eigen::Vector3d rotateSingleVector(const Eigen::Vector3d& vec, const Eigen::Vector3d& axis, double angle)
{
	Eigen::Vector3d rotVec;

	// first normalize axis
	double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
	Eigen::Matrix3d rotMat;

	double c = std::cos(angle);
	double s = std::sin(angle);
	rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
		uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
		uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

	rotVec = rotMat * vec;
	return rotVec;
}

void rotateIntrinsicVector(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const std::vector<RotateVertexInfo>& rotVerts, Eigen::MatrixXd& rotHalfEdgeW)
{
	Eigen::MatrixXd vertNormals;
	igl::per_vertex_normals(V, mesh.faces(), vertNormals);

	Eigen::MatrixXd vertVec = intrinsicHalfEdgeVec2VertexVec(halfEdgeW, V, mesh);

	for (int i = 0; i < rotVerts.size(); i++)
	{
		int vid = rotVerts[i].vid;
		vertVec.row(vid) = rotateSingleVector(vertVec.row(vid), vertNormals.row(vid), rotVerts[i].angle);
	}

	rotHalfEdgeW = vertexVec2IntrinsicHalfEdgeVec(vertVec, V, mesh);
}

void buildVertexNeighboringInfo(const MeshConnectivity& mesh, int nverts, std::vector<std::vector<int>>& vertNeiEdges, std::vector<std::vector<int>>& vertNeiFaces)
{
    vertNeiEdges.resize(nverts);
    vertNeiFaces.resize(nverts);

    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();

    for(int i = 0; i < nfaces; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            vertNeiFaces[mesh.faceVertex(i, j)].push_back(i);
        }
    }

    for(int i = 0; i < nedges; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            vertNeiEdges[mesh.edgeVertex(i, j)].push_back(i);
        }
    }

}


Eigen::SparseMatrix<double> buildD0(const MeshConnectivity& mesh, int nverts)
{
    int nedges = mesh.nEdges();

    std::vector<Eigen::Triplet<double>> T;

    for(int i = 0; i < nedges; i++)
    {
        T.push_back(Eigen::Triplet<double>(i, mesh.edgeVertex(i, 1), 1));
        T.push_back(Eigen::Triplet<double>(i, mesh.edgeVertex(i, 0), -1));
    }

    Eigen::SparseMatrix<double> M;
    M.resize(nedges, nverts);
    M.setFromTriplets(T.begin(), T.end());
    return M;
}

void mkdir(const std::string& foldername)
{
    if (!std::filesystem::exists(foldername))
    {
        std::cout << "create directory: " << foldername << std::endl;
        if (!std::filesystem::create_directory(foldername))
        {
            std::cerr << "create folder failed." << foldername << std::endl;
            exit(1);
        }
    }
}

Eigen::VectorXd getFaceArea(const Eigen::MatrixXd& V, const MeshConnectivity& mesh)
{
	Eigen::VectorXd faceArea;
	igl::doublearea(V, mesh.faces(), faceArea);
	faceArea /= 2;
	return faceArea;
}

Eigen::VectorXd getEdgeArea(const Eigen::MatrixXd& V, const MeshConnectivity& mesh)
{
	Eigen::VectorXd faceArea = getFaceArea(V, mesh);
	Eigen::VectorXd edgeArea;
	edgeArea.setZero(mesh.nEdges());

	for (int i = 0; i < mesh.nEdges(); i++)
	{
		int f0 = mesh.edgeFace(i, 0);
		int f1 = mesh.edgeFace(i, 1);

		if (f0 != -1 && f1 != -1)
			edgeArea(i) = (faceArea(f0) + faceArea(f1)) / 2.;
		else if (f0 == -1)
			edgeArea(i) = faceArea(f1) / 2.;
		else
			edgeArea(i) = faceArea(f0) / 2.;
	}
	return edgeArea;
}


Eigen::VectorXd getVertArea(const Eigen::MatrixXd& V, const MeshConnectivity& mesh)
{
	Eigen::VectorXd faceArea = getFaceArea(V, mesh);
	Eigen::VectorXd vertArea;
	vertArea.setZero(V.rows());

	for (int i = 0; i < mesh.nFaces(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);
			vertArea(vid) += faceArea(i) / 3.;
		}
	}

	return vertArea;
}

void laplacianSmoothing(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& newV, double smoothingRatio, int opTimes)
{
    newV = V;
    if(opTimes == 0)
        return;

    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);

    Eigen::VectorXd sum;
    sum.setZero(V.rows());

    for (int k=0; k<L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it)
        {
            if(it.row() == it.col())
                sum(it.row()) = it.value();
        }

    std::vector<Eigen::Triplet<double>> T;

    for (int k=0; k<L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it)
        {
            if(sum(it.row()) != 0)
                T.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value() / sum(it.row())));
            else
                T.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value() / (1e-6 + sum(it.row()))));
        }

    L.setFromTriplets(T.begin(), T.end());      // normalized laplacian
    Eigen::SparseMatrix<double> idmat(V.rows(), V.rows());
    idmat.setIdentity();

    Eigen::SparseMatrix<double> smoothL = idmat - smoothingRatio * L;

    std::vector<int> bnd;
    igl::boundary_loop(F, bnd);

    for(int i = 0; i < opTimes; i++)
    {
        for(int j = 0; j < 3; j++)
            newV.col(j) = smoothL * newV.col(j);

        for(int j = 0; j < bnd.size(); j++)
            newV.row(bnd[j]) = V.row(bnd[j]);
    }


}

void laplacianSmoothing(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& oldData, Eigen::VectorXd& newData, double smoothingRatio, int opTimes)
{
	newData = oldData;
	if (opTimes == 0)
		return;

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(V, F, L);

	Eigen::VectorXd sum;
	sum.setZero(V.rows());

	for (int k = 0; k < L.outerSize(); ++k)
		for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
		{
			if (it.row() == it.col())
				sum(it.row()) = it.value();
		}

	std::vector<Eigen::Triplet<double>> T;

	for (int k = 0; k < L.outerSize(); ++k)
		for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
		{
			if (sum(it.row()) != 0)
				T.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value() / sum(it.row())));
			else
				T.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value() / (1e-6 + sum(it.row()))));
		}

	L.setFromTriplets(T.begin(), T.end());      // normalized laplacian
	Eigen::SparseMatrix<double> idmat(V.rows(), V.rows());
	idmat.setIdentity();

	Eigen::SparseMatrix<double> smoothL = idmat - smoothingRatio * L;

	for (int i = 0; i < opTimes; i++)
	{
		newData = smoothL * newData;
	}
}

void curvedPNTriangleUpsampling(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& VN, const std::vector<std::pair<int, Eigen::Vector3d>>& baryList, Eigen::MatrixXd& NV, Eigen::MatrixXd& newVN)
{
	int nupverts = baryList.size();
	NV.setZero(nupverts, 3);
	newVN.setZero(nupverts, 3);

	for (int i = 0; i < nupverts; i++)
	{
		int fid = baryList[i].first;

		double w = baryList[i].second(0);
		double u = baryList[i].second(1);
		double v = baryList[i].second(2);

		Eigen::Vector3d P1 = V.row(F(fid, 0));
		Eigen::Vector3d P2 = V.row(F(fid, 1));
		Eigen::Vector3d P3 = V.row(F(fid, 2));

		Eigen::Vector3d N1 = VN.row(F(fid, 0));
		Eigen::Vector3d N2 = VN.row(F(fid, 1));
		Eigen::Vector3d N3 = VN.row(F(fid, 2));

		N1 = N1 / N1.norm();
		N2 = N2 / N2.norm();
		N3 = N3 / N3.norm();

		Eigen::Vector3d b300 = P1;
		Eigen::Vector3d b030 = P2;
		Eigen::Vector3d b003 = P3;

		double w12 = (P2 - P1).dot(N1);
		double w21 = (P1 - P2).dot(N2);
		double w23 = (P3 - P2).dot(N2);
		double w32 = (P2 - P3).dot(N3);
		double w31 = (P1 - P3).dot(N3);
		double w13 = (P3 - P1).dot(N1);

		Eigen::Vector3d b210 = (2 * P1 + P2 - w12 * N1) / 3;
		Eigen::Vector3d b120 = (2 * P2 + P1 - w21 * N2) / 3;
		Eigen::Vector3d b021 = (2 * P2 + P3 - w23 * N2) / 3;
		Eigen::Vector3d b012 = (2 * P3 + P2 - w32 * N3) / 3;
		Eigen::Vector3d b102 = (2 * P3 + P1 - w31 * N3) / 3;
		Eigen::Vector3d b201 = (2 * P1 + P3 - w12 * N1) / 3;

		Eigen::Vector3d Ep = (b210 + b120 + b021 + b012 + b102 + b201) / 6;
		Eigen::Vector3d Vp = (P1 + P2 + P3) / 3;
		Eigen::Vector3d b111 = Ep + (Ep - Vp) / 2;

		NV.row(i) = b300 * w * w * w + b030 * u * u * u + b003 * v * v * v
			+ b210 * 3 * w * w * u + b120 * 3 * w * u * u + b201 * 3 * w * w * v
			+ b021 * 3 * u * u * v + b102 * 3 * w * v * v + b012 * 3 * u * v * v
			+ b111 * 6 * w * u * v;

		Eigen::Vector3d n200 = N1;
		Eigen::Vector3d n020 = N2;
		Eigen::Vector3d n002 = N3;

		double v12 = 2 * (P2 - P1).dot(N1 + N2) / (P2 - P1).dot(P2 - P1);
		double v23 = 2 * (P3 - P2).dot(N2 + N3) / (P3 - P2).dot(P3 - P2);
		double v31 = 2 * (P1 - P3).dot(N3 + N1) / (P1 - P3).dot(P1 - P3);

		Eigen::Vector3d n110 = N1 + N2 - v12 * (P2 - P1);
		Eigen::Vector3d n011 = N2 + N3 - v23 * (P3 - P2);
		Eigen::Vector3d n101 = N3 + N1 - v31 * (P1 - P3);

		n110 = n110 / n110.norm();
		n011 = n011 / n011.norm();
		n101 = n101 / n101.norm();

		newVN.row(i) = n200 * w * w + n020 * u * u + n002 * v * v + n110 * w * u + n011 * u * v + n101 * w * v;
	}
}

void getWrinkledMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<std::complex<double>>& zvals, std::vector<std::vector<int>>* vertNeiFaces, Eigen::MatrixXd& wrinkledV, double scaleRatio, bool isTangentCorrection)
{
	int nverts = V.rows();
	int nfaces = F.rows();

	if(!vertNeiFaces)
	{ 
		std::vector<std::vector<int>> vertNeiEdges;
		buildVertexNeighboringInfo(MeshConnectivity(F), nverts, vertNeiEdges, *vertNeiFaces);
	}
	
	wrinkledV = V;
	Eigen::MatrixXd VN;
	igl::per_vertex_normals(V, F, VN);

	for (int vid = 0; vid < nverts; vid++)
	{
		wrinkledV.row(vid) += scaleRatio * (zvals[vid].real() * VN.row(vid));
	}

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = F(i, j);
			Eigen::Matrix2d Ib;
			Eigen::Matrix<double, 3, 2> drb;
			drb.col(0) = (V.row(F(i, (j + 1) % 3)) - V.row(F(i, j))).transpose();
			drb.col(1) = (V.row(F(i, (j + 2) % 3)) - V.row(F(i, j))).transpose();

			Ib = drb.transpose() * drb;
			
			std::complex<double> dz0 = zvals[F(i, (j + 1) % 3)] - zvals[F(i, j)];
			std::complex<double> dz1 = zvals[F(i, (j + 2) % 3)] - zvals[F(i, j)];

			Eigen::Vector2d aSqdtheta;
			aSqdtheta << (std::conj(zvals[vid]) * dz0).imag(), (std::conj(zvals[vid]) * dz1).imag();

			Eigen::Vector3d extASqdtheta = drb * Ib.inverse() * aSqdtheta;

			double theta = std::arg(zvals[vid]);

			if (isTangentCorrection)
			{
				wrinkledV.row(vid) += scaleRatio / vertNeiFaces->at(vid).size() * (1. / 8 * std::sin(2 * theta) * extASqdtheta);
			}

			
		}
	}
}

void computeBaryGradient(const Eigen::Vector3d& P0, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, const Eigen::Vector3d& bary, Eigen::Matrix3d& baryGrad)
{
	//P = bary(0) * P0 + bary(1) * P1 + bary(2) * P2;
	Eigen::Matrix2d I, Iinv;

	I << (P1 - P0).squaredNorm(), (P1 - P0).dot(P2 - P0), (P2 - P0).dot(P1 - P0), (P2 - P0).squaredNorm();
	Iinv = I.inverse();

	Eigen::Matrix<double, 3, 2> dr;
	dr.col(0) = P1 - P0;
	dr.col(1) = P2 - P0;

	Eigen::Matrix<double, 2, 3> dbary12 = Iinv * dr.transpose();

	baryGrad.row(0) = -dbary12.row(0) - dbary12.row(1);
	baryGrad.row(1) = dbary12.row(0);
	baryGrad.row(2) = dbary12.row(1);
}
