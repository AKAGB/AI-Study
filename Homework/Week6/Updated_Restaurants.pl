restaurant(mixuebingcheng,1998,drinks).
restaurant(muwushaokao,2003,barbecue).
restaurant(diandude,1993,yuecai).
restaurant(ajukejiacai,2007,yuecai).
restaurant(hongmenyan,2015,yuecai).
restaurant(dagangxianmiaoshaoji,2015,yuecai).
restaurant(huangmenjimifan,1935,lucai).
restaurant(shaxianxiaochi,1998,mincai).
restaurant(tongxianghui,2013,xiangcai).
restaurant(yangguofu,2007,dongbeicai).

branch(mixuebingcheng,wushan).
branch(mixuebingcheng,lujiang).
branch(mixuebingcheng,shipaixi).
branch(mixuebingcheng,yiyuannan).
branch(mixuebingcheng,beiting).
branch(mixuebingcheng,xintiandi).
branch(mixuebingcheng,beigang).
branch(mixuebingcheng,chentian).
branch(mixuebingcheng,chisha).
branch(mixuebingcheng,longdong).
branch(mixuebingcheng,zhucun).
branch(mixuebingcheng,shiqiao).

branch(muwushaokao,gangding).
branch(muwushaokao,shayuan).
branch(muwushaokao,heguang).
branch(muwushaokao,tangxia).
branch(muwushaokao,dongpu).
branch(muwushaokao,shengdi).
branch(muwushaokao,xiaogang).
branch(muwushaokao,tonghe).
branch(muwushaokao,diwangguangchang).
branch(muwushaokao,runzhengguangchang).

branch(diandude,huachengdadao).
branch(diandude,zhongshansi).
branch(diandude,huifudong).
branch(diandude,youtuobangshiguang).
branch(diandude,bainaohui).
branch(diandude,panfu).
branch(diandude,yangji).
branch(diandude,tianhebei).
branch(diandude,shiqiao).
branch(diandude,linhe).

branch(ajukejiacai,yongfu).
branch(ajukejiacai,xintiandi).
branch(ajukejiacai,shatainan).

branch(hongmenyan,xintiandi).
branch(hongmenyan,zhilanwan).

branch(dagangxianmiaoshaoji,yuancun).
branch(dagangxianmiaoshaoji,kecun).
branch(dagangxianmiaoshaoji,beishan).
branch(dagangxianmiaoshaoji,nanpudadao).
branch(dagangxianmiaoshaoji,xinshi).
branch(dagangxianmiaoshaoji,dongpu).
branch(dagangxianmiaoshaoji,huadong).
branch(dagangxianmiaoshaoji,fangcun).
branch(dagangxianmiaoshaoji,cencun).
branch(dagangxianmiaoshaoji,changxing).
branch(dagangxianmiaoshaoji,gaosheng).

branch(huangmenjimifan,siyoubei).
branch(huangmenjimifan,yuancun).
branch(huangmenjimifan,dongxiaonan).
branch(huangmenjimifan,dongxiaonan).
branch(huangmenjimifan,dongqu).
branch(huangmenjimifan,dalingang).
branch(huangmenjimifan,pazhou).
branch(huangmenjimifan,beigang).

branch(shaxianxiaochi,kangwangnan).
branch(shaxianxiaochi,beigang).
branch(shaxianxiaochi,luolang).

branch(yangguofu,xintiandi).
branch(yangguofu,dayuan).
branch(yangguofu,shishangtianhe).
branch(yangguofu,chebei).

branch(tongxianghui,bainaohui).
branch(tongxianghui,tianhebei).
branch(tongxianghui,yongfu).
branch(tongxianghui,shimaocheng).
branch(tongxianghui,hanting).
branch(tongxianghui,yuanyangmingyuan).
branch(tongxianghui,zhongshanyilu).
branch(tongxianghui,huizhoudasha).
branch(tongxianghui,kaifadadao).
branch(tongxianghui,maoshengdasha).

district(wushan,tianhe).
district(shipaixi,tianhe).
district(longdong,tianhe).
district(gangding,tianhe).
district(heguang,tianhe).
district(tangxia,tianhe).
district(dongpu,tianhe).
district(huachengdadao,tianhe).
district(youtuobangshiguang,tianhe).
district(bainaohui,tianhe).
district(tianhebei,tianhe).
district(linhe,tianhe).
district(yuancun,tianhe).
district(cencun,tianhe).
district(changxing,tianhe).
district(dalingang,tianhe).
district(shishangtianhe,tianhe).
district(chebei,tianhe).
district(bainaohui,tianhe).
district(hanting,tianhe).
district(yuanyangmingyuan,tianhe).


district(lujiang,haizhu).
district(yiyuannan,haizhu).
district(chisha,haizhu).
district(shayuan,haizhu).
district(xiaogang,haizhu).
district(runzhengguangchang,haizhu).
district(kecun,haizhu).
district(beishan,haizhu).
district(dongxiaonan,haizhu).
district(pazhou,haizhu).
district(huizhoudasha,haizhu).

district(beiting,panyu).
district(beigang,panyu).
district(xintiandi,panyu).
district(shiqiao,panyu).
district(zhilanwan,panyu).
district(nanpudadao,panyu).
district(maoshengdasha,panyu).

district(chentian,baiyun).
district(shengdi,baiyun).
district(tonghe,baiyun).
district(shatainan,baiyun).
district(xinshi,baiyun).
district(dayuan,baiyun).

district(zhucun,huadu).
district(huadong,huadu).

district(diwangguangchang,yuexiu).
district(zhongshansi,yuexiu).
district(huifudong,yuexiu).
district(panfu,yuexiu).
district(yangji,yuexiu).
district(yongfu,yuexiu).
district(siyoubei,yuexiu).
district(zhongshanyilu,yuexiu).

district(fangcun,liwan).
district(gaosheng,liwan).
district(kangwangnan,liwan).
district(shimaocheng,liwan).

district(dongqu,huangpu).
district(luolang,huangpu).
district(kaifadadao,huangpu).

% Question1
answer1(Ans):-setof(X,branch(X,beigang),Ans).

% Question2
district_yuexiang(X):-restaurant(Z,_,yuecai),restaurant(M,_,xiangcai),branch(Z,Y),branch(M,N),district(Y,X),district(N,X).
answer2(Ans):-setof(X,district_yuexiang(X),Ans).

% Question3
% return minmum of list.
find_min([X],X).
find_min(List, Minimum):- List = [Head|Tail],
    find_min(Tail, TailMin),
    (Head < TailMin ->
       Minimum is Head;
       Minimum is TailMin
    ).

% Cnt is number of branches of Rest.
branches_of_rest(Rest, Cnt):-restaurant(Rest,_,_),findall(X, branch(Rest,X), List),length(List, Cnt).
% return all branch.
all_branches(Bran):-findall(X, (restaurant(Rs,_,_),branches_of_rest(Rs, X)) ,Bran).

answer3(Ans):-all_branches(Bs),find_min(Bs, MinBranches),findall(X, branches_of_rest(X, MinBranches), Ans).

% Question4
restes_of_area(Area, Cnt):-branch(_,Area),findall(X, branch(X, Area), Ans),length(Ans, Cnt).

answer4(Ans):-findall(X, (branch(_,X),restes_of_area(X,K),K>1), List),setof(X, member(X,List), Ans).

% Question5
allRestYear(T):-findall(X, restaurant(_,X,_), T).
answer5(Ans):-allRestYear(Years),find_min(Years, MinYear),findall(X,restaurant(X,MinYear,_),Ans).

% Question6
answer6(Ans):-findall(X, (branches_of_rest(X, K), K >= 10), Ans).

sameDistrict(Rest1, Rest2):-branch(Rest1, Area1),branch(Rest2, Area2),Rest1\=Rest2,district(Area1, Z),district(Area2, Z).
% Test
test(Ans):-setof([X,Y], sameDistrict(X,Y), Ans).