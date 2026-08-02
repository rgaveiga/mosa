# Plano de implementação: restart baseado no archive

## Objetivo

Reduzir o wall time de `Anneal.evolve()` sem alterar suas convenções públicas.
O archive passa a ser simultaneamente o resultado Pareto e a fonte persistida
para um warm start após uma interrupção. Novos checkpoints deixam de ser
produzidos.

O restart é um warm start, não a continuação exata da cadeia interrompida. O
checkpoint anterior também não persistia temperatura, iteração, estado do RNG,
contador de rejeições ou estado de seleção.

## Modelo de recuperação

- `restart=False`: limpa o archive em memória e inicia aleatoriamente.
- `restart=True`: carrega o archive e usa seu último par `x/f` retido.
- A aplicação configura população e parâmetros antes de `evolve()`, como já
  ocorre nos exemplos do projeto.
- Um `checkpoint.json` legado continua legível como fallback de migração, mas
  nenhum novo checkpoint é gravado.
- Archive ausente resulta em inicialização aleatória.
- Archive incompatível com a população configurada gera `MOSAError`.

A última entrada não precisa coincidir com o estado interrompido: ela é a
solução não dominada retida mais recentemente e constitui um ponto inicial
melhor que uma solução aleatória.

## Reconstrução da população

O archive fornece `x/f`; a aplicação fornece novamente a população original.

- Contínuo: reutilizar limites e validar os valores recuperados.
- Discreto não distinto: reutilizar toda a população e associar os valores
  recuperados às entradas configuradas.
- Discreto distinto: copiar a população e subtrair uma ocorrência por valor da
  solução recuperada.
- Tamanho variável: derivar o tamanho corrente da solução arquivada, preservando
  sua representação pública.
- Duplicatas/categorias: usar a igualdade semântica de `_GroupState`, mantendo
  cardinalidade de multiconjunto e identidade dos objetos configurados quando
  possível.

## Política de persistência

Construir o archive em memória e gravá-lo somente quando estiver dirty:

1. após a primeira temperatura concluída;
2. a cada `archive_save_interval` temperaturas;
3. no término normal; e
4. antes do retorno antecipado por excesso de rejeições.

O padrão é 10 temperaturas. `0` significa somente no final e `1` solicita o
comportamento conservador por temperatura. Um gatilho por tempo pode ser
adicionado depois para funções objetivo excepcionalmente demoradas.

## Escrita durável

`savex()` e a persistência automática devem compartilhar o mesmo caminho:

1. criar um snapshot público consistente;
2. serializar JSON compacto;
3. gravar arquivo temporário no diretório de destino;
4. executar `flush()` e `os.fsync()`;
5. manter a geração anterior como `<archive>.bak`;
6. substituir o primário com `os.replace()`; e
7. remover temporário abandonado em caso de erro.

O carregamento tenta primeiro o primário e depois o backup.

## Compatibilidade

- Preservar `evolve(func)`, `restart`, `archive_file`, `savex`, `loadx` e o
  formato `{"x": ..., "f": ...}`.
- Preservar ordem e quantidade das chamadas aleatórias.
- Não alterar argumentos da função objetivo nem representações públicas.
- Ler checkpoints legados durante a migração.
- `checkpoint_interval` surgiu apenas no branch de performance e deve ser
  removido antes de ser incorporado à API legada.

## Sequência de implementação

1. Criar loader de archive com fallback para backup.
2. Selecionar a última entrada quando restart e população estiverem disponíveis.
3. Reconstruir e validar pools discretos nos estados internos.
4. Usar checkpoint legado somente quando o restart pelo archive não for viável.
5. Remover todas as gravações de checkpoint de `evolve()`.
6. Definir intervalo padrão 10 e salvar no primeiro/periódico/final.
7. Substituir `json.dump(open(...))` por escrita atômica e durável.
8. Atualizar mensagens e documentação de restart.
9. Adicionar testes funcionais, de recuperação, corrupção e cadência.
10. Rodar a suíte determinística e medir somente o corpo de `evolve()`.

## Testes obrigatórios

- Restart usa a última entrada armazenada.
- Populações contínua, discreta, categórica, distinta, duplicada e variável são
  reconstruídas corretamente.
- Archive incompatível gera `MOSAError`.
- Checkpoint legado é aceito quando necessário.
- Nenhum novo `checkpoint.json` é criado.
- Com padrão 10, saves dirty ocorrem nas temperaturas 1, 10, 20 e final.
- Intervalos `0` e `1` mantêm suas semânticas.
- Término antecipado persiste archive dirty.
- Primário corrompido carrega `.bak`.
- Escrita não deixa temporário nem handle aberto no Windows.
- Resultados seeded e assinaturas públicas permanecem estáveis.

## Critério de performance

Medir `perf_counter()` e `process_time()` imediatamente ao redor de `evolve()`.
Imports e preparação ficam fora da medição.

| Política | Wall time | CPU time |
| --- | ---: | ---: |
| checkpoint/archive por temperatura | 5,920 s | 0,234 s |
| archive somente ao final | 0,056 s | 0,031 s |
| archive a cada 10 temperaturas | 0,160 s | 0,031 s |
| implementação durável a cada 10 temperaturas | 0,403 s | 0,063 s |

A última linha mede a implementação com JSON compacto, `fsync`, backup e
substituição atômica no diretório sincronizado. Ela mantém a janela de
recuperação e reduz o wall time em aproximadamente 14,7 vezes.

## Otimizações posteriores

Depois que persistência e restart estiverem corretos e medidos:

- caminho `O(1)` para archive com um objetivo;
- comparação Pareto em fases, evitando máscara desnecessária após rejeição;
- buffers booleanos reutilizáveis;
- escalares Python e representação híbrida para grupos dinâmicos; e
- menos lookups e overhead de `numpy.random.choice` no loop interno.
