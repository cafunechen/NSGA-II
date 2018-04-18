% Implementação NSGA-II (Non-dominated Sorting Genetic Algorithm II)
% Binário
% Autor: Thiago Silva
function NSGAII_Bin()
Npop = 100;
TempNpop = Npop*2;
Nvar = 20;
Nobj = 2;
Ngen = 100;
Pc = 0.9;
Pm = 1/Nvar;

% Limites do problema
xmax = 1000;
xmin = -1000;
Info = Nvar+Nobj+1;

% População inicial
Pop = IniPop(Npop, Nvar, Info);
SelPop = zeros(Npop, Info);
TempPop = zeros(TempNpop, Info);

% Problema de minimização
Dom = @(a, b) Dominate(a, b, Nvar, Nobj);

CDist = @(Pop, Npop) CrowdingDistance(Pop, Npop, Nvar, Nobj, xmax, xmin);

% Fitness População
for K=1:Npop
    Pop(K, Nvar+1:Info) = Fitness(Pop(K, :), Nvar, xmax, xmin);
end

% Gerações
for gen=1:Ngen
    disp(['Iteração ' num2str(gen)])
    
    % Construção da próxima geração
    TempPop(1:Npop, :) = Pop;
    
    % Seleção
    F = FastNonDominatedSort(Pop, Npop, Dom);
    for K=1:Npop
        SelIndex = Torneio(F, Pop, Npop, CDist);
        SelPop(K, :) = Pop(SelIndex, :);
    end
    
    % Randomiza seleção
    Rands = randperm(Npop);
    
    % Cruzamento
    K = 1;
    while K <= Npop
        if rand() < Pc
            Pai1 = SelPop(Rands(K), :);
            Pai2 = SelPop(Rands(mod(K, Npop) + 1), :);
            [Filho1, Filho2] = CruzamentoUniforme(Pai1, Pai2, Nvar);
            if (Npop-K) > 1
                Filho1(Nvar+1:Info) = Fitness(Filho1, Nvar, xmax, xmin);
                Filho2(Nvar+1:Info) = Fitness(Filho2, Nvar, xmax, xmin);
                TempPop(Npop+K, :) = Filho1;
                TempPop(Npop+K+1, :) = Filho2;
                K = K + 2;
            else
                if rand() < 0.5
                    Filho1(Nvar+1:Info) = Fitness(Filho1, Nvar, xmax, xmin);
                    TempPop(Npop+K, :) = Filho1;
                else
                    Filho2(Nvar+1:Info) = Fitness(Filho2, Nvar, xmax, xmin);
                    TempPop(Npop+K, :) = Filho2;
                end
                K = K + 1;
            end
        end
    end
    
    % Mutação
    K = 1;
    while K <= Npop
        if rand() < Pm
            X = SelPop(Rands(K), :);
            X = MutacaoBit(X, Nvar);
            X(Nvar+1:Info) = Fitness(X, Nvar, xmax, xmin);
            TempPop(Npop+K, :) = X;
        end
        K = K + 1;
    end
    
    % Atualiza fronteiras Pareto
    F = FastNonDominatedSort(TempPop, TempNpop, Dom);
    Findex = 1;
    NpopAtual = 0;
    while true
        Temp = TempPop(F == Findex, :);
        Ntemp = size(Temp, 1);
        
        if (NpopAtual+Ntemp) >= Npop
            break
        end
        
        Pop(NpopAtual+1:NpopAtual+Ntemp, :) = Temp;
        NpopAtual = NpopAtual + Ntemp;
        Findex = Findex + 1;
    end
    
    % População com próxima fronteira
    while NpopAtual < Npop
        Temp = TempPop(F == Findex, :);
        Ntemp = size(Temp, 1);
        
        if Ntemp == 1
            Pop(NpopAtual+1, :) = Temp(1, :);
        else
            Index = randperm(Ntemp, 2);
            Dist = CDist(Temp, Ntemp);
            if Dist(Index(1)) > Dist(Index(2))
                Pop(NpopAtual+1, :) = Temp(Index(1), :);
            else
                Pop(NpopAtual+1, :) = Temp(Index(2), :);
            end
        end
        NpopAtual = NpopAtual + Ntemp;
    end
end

% Divisão em fronteiras
F = FastNonDominatedSort(Pop, Npop, Dom);

plot(Pop(F == 1, Nvar+1), Pop(F == 1, Nvar+2), 'r*')
title(['Iteração ' num2str(Ngen)])
%hold on
grid on
% plot(Pop(F == 2, Nvar+1), Pop(F == 2, Nvar+2), 'bo')
% plot(Pop(F == 3, Nvar+1), Pop(F == 3, Nvar+2), 'go')
% plot(Pop(F == 4, Nvar+1), Pop(F == 4, Nvar+2), 'ko')
end

%%
% Verifica se o vetor A domina o vetor B.
%  Se R = 1,  A domina B
%  Se R = -1, B domina A
%  Senão A e B são incomparáveis
function R = Dominate(A, B, Nvar, Nobj)
I = Nvar+1;
J = Nvar+Nobj;
if all(A(I:J) <= B(I:J)) && any(A(I:J) < B(I:J))
    R = 1;
elseif  all(A(I:J) >= B(I:J)) && any(A(I:J) > B(I:J))
    R = -1;
else
    R = 0;
end
end


% Função-custo (binário)
function R = Fitness(Cromo, Nvar, xmax, xmin)
i = Cromo(1:Nvar);
s = sum(2.^(Nvar-1:-1:0) .* i);
x = xmin + (xmax-xmin)/(2^Nvar-1) * s;
R = zeros(1, 2);
R(1) = x.^2;
R(2) = (x-2).^2;
R(3) = x;
end


% Inicaliza população
function P = IniPop(Npop, Nvar, info)
P = zeros(Npop, info);
for i=1:Npop
    for j=1:Nvar
        P(i, j) = round(rand());
    end
end
end

%%
% Distáncia de multidão
function Dist = CrowdingDistance(P, NP, Nvar, Nobj, xmax, xmin)
Dist = zeros(NP, 1);
for M=1:Nobj
    [~, Pos] = sort(P(:, Nvar+M));
    Dist(Pos(1)) = Inf;
    Dist(Pos(end)) = Inf;
    for K=2:NP-1
        Dist(Pos(K)) = Dist(Pos(K)) + abs((P(Pos(K+1), Nvar+M) - P(Pos(K-1), Nvar+M)) / abs(xmax - xmin));
    end
end
end

%%
% Torneio Binário
function SelIndex = Torneio(F, Pop, Npop, CDist)
Index = randperm(Npop, 2);
if F(Index(1)) < F(Index(2))
    SelIndex = Index(1);
elseif F(Index(1)) > F(Index(2))
    SelIndex = Index(2);
else
    Pos = find(F == F(Index(1)));
    Temp = Pop(Pos, :);
    Ntemp = size(Temp, 1);
    
    Dist = CDist(Temp, Ntemp);
    
    Pa = (Pos == Index(1));
    Pb = (Pos == Index(2));
    if Dist(Pa) > Dist(Pb)
        SelIndex = Index(1);
    else
        SelIndex = Index(2);
    end
end
end

%%
% Cruzamento Uniforme
function [ F1, F2 ] = CruzamentoUniforme(Pai1, Pai2, Nvar)
F1 = Pai1;
F2 = Pai2;
Mascara = round(rand(Nvar, 1));
for i=1:Nvar
    if Mascara(i) == 1
        F1(i) = Pai2(i);
        F2(i) = Pai1(i);
    end
end
end

%%
% Mutação Bit
function M = MutacaoBit( X, Nvar )
M = X;
Index = randi(Nvar);
for i=1:Index
    M(i) = ~M(i);
end
end

%%
% A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
% Retorna a fronteira Pareto
function F = FastNonDominatedSort(Pop, Npop, Dom)
F = zeros(1, Npop);
S = cell(1, Npop);
N = zeros(1, Npop);

for i=1:Npop
    p = Pop(i, :);
    for j=1:Npop
        q = Pop(j, :);
        D = Dom(p, q);
        if  D == 1
            S{i} = [S{i} j];
        elseif D == -1;
            N(i) = N(i) + 1;
        end
    end
    if N(i) == 0
        F(i) = 1;
    end
end

i=1;
FP = find(F == 1);
while ~isempty(FP)
    for p=FP
        for q=S{p}
            N(q) = N(q) - 1;
            if N(q) == 0
                F(q) = i + 1;
            end
        end
    end
    i = i + 1;
    FP = find(F == i);
end
end
