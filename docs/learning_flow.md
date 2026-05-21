# Red Bot Learning Flow

Doel: de bot leert autonoom, test autonoom en vraagt alleen operatorbesluiten wanneer een voorstel live gedrag of duidelijke operator-aandacht raakt.

## Effect Levels

### context_live

Autonome learning die GPT-context, coin profiles of marktcontext verbetert. Dit heeft indirect live invloed, omdat GPT betere context krijgt, maar wijzigt geen harde orderregels.

Voorbeelden:
- coin profile update
- indicator edge als context
- market regime context
- coin-specifieke observaties voor GPT

Default: autonoom toegestaan, zichtbaar als verwerkt contextbewijs.

### shadow_only

Autonome simulatie zonder live effect.

Voorbeelden:
- risk-down replay
- pre-GPT gate replay
- exit-rule replay
- guard threshold replay

Default: altijd autonoom draaien. Operator beslist pas over de uitkomst als die naar een hoger effect level wil.

### risk_down_live

Live effect dat alleen risico verlaagt.

Voorbeelden:
- size omlaag
- geen nieuwe longs in risk-off
- tijdelijke entry-pauze

Default: operator approval vereist. Later kan dit semi-autonoom worden als bewijs, safety en audit volwassen genoeg zijn.

### strategy_live

Live effect op entry, exit, TP/SL, sizing-regels of indicator thresholds.

Default: strengste approval. Niet automatisch.

### risk_up_live

Live effect dat risico verhoogt.

Default: geblokkeerd tot expliciet nieuw ontwerp.

## Operator Actions

### approve

Akkoord met de volgende veilige fase voor het effect level.

- context_live: mag als live context verwerkt blijven.
- shadow_only: mag als shadow learning blijven lopen.
- risk_down_live: pending live gate; geen live effect zonder safety/live wiring.
- strategy_live: pending strict live gate; geen live effect zonder later ontwerp.

### wait

Blijf volgen en verzamel meer bewijs. De recommendation mag later terugkomen met sterker bewijs.

### reject

Niet opnieuw pushen, tenzij er wezenlijk nieuw bewijs is.

### freeze

Onderwerp tijdelijk parkeren. De bot mag intern blijven meten, maar moet dit onderwerp niet blijven aandragen als operatorbesluit.

### snooze

Tijdelijk verbergen tot expiry.

### note

Operatorcontext toevoegen zonder statusbesluit.

## Pipeline

1. Observe: events, candles, trades, GPT decisions en outcomes verzamelen.
2. Label: outcomes en counterfactuals labelen.
3. Analyze: ML, indicator edge, exits, risk bridge, guards en readiness rapporteren.
4. Propose: recommendation aggregator maakt compacte decision cards.
5. Resolve: operator decision resolver past eerdere approve/reject/wait/freeze toe.
6. Review: app toont alleen actieve, relevante kaarten.
7. Effect: alleen context_live en shadow_only mogen autonoom; live controls vereisen gates.

## App Principle

De app toont geen rapportenbak. De app toont:

- dagelijkse status
- actieve review cards
- onderbouwing per kaart
- effect level
- wat er gebeurt na klikken
- trade/position context
- audit history
