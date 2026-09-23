const form = document.querySelector('#request-form');
const results = document.querySelector('#results');
const statusBox = document.querySelector('#status');
const submit = document.querySelector('#submit');
const trace = document.querySelector('#trace');
const timing = document.querySelector('#timing');
const panel = document.querySelector('.results-panel');
const money = new Intl.NumberFormat('ru-RU');
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let activeRequest;

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : 'Проверьте параметры: бюджет не может быть отрицательным, длительность должна быть больше нуля.');
  }
  return data;
}
function fill(query) {
  for (const element of form.elements) if (element.name) element.value = query[element.name] ?? '';
}
function showResponse(data) {
  const headings = {FOUND:'Подобрали варианты',NO_CATEGORY:'В этом городе такой категории нет',NO_MATCH:'Кандидаты есть, но никто не подходит'};
  statusBox.innerHTML = data.message ? `<div class="notice">${escape(data.message)}</div>` : '';
  if (!data.cards.length) {
    results.innerHTML = `<div class="empty"><div class="empty-symbol" aria-hidden="true">○</div><h3>${headings[data.outcome]}</h3><p>Попробуйте изменить параметры — условия не будут изменены автоматически.</p></div>`;
  } else {
    results.innerHTML = `<div class="result-summary">${headings[data.outcome]} · показано ${data.found_count} из ${data.eligible_count} подходящих</div>` + data.cards.map((card,index) => `
      <article class="card"><div class="card-head"><span class="rank">${String(index+1).padStart(2,'0')}</span><div><h3>${escape(card.name)}</h3><div class="location">${escape(card.category)} · ${escape(card.city)}</div></div><div class="price">от ${money.format(card.price_from_kzt)} ₸<small>за мероприятие</small></div></div>
      <p class="explanation">${escape(card.explanation)}</p>
      ${(card.price_imputed || card.city_imputed) ? `<p class="data-note">${[card.price_imputed ? 'Цена проставлена при подготовке датасета' : '',card.city_imputed ? 'Город проставлен при подготовке датасета' : ''].filter(Boolean).join(' · ')}</p>` : ''}
      <div class="card-bottom"><span class="tag ${card.is_synthetic ? 'synthetic':''}">${card.is_synthetic ? 'Синтетический профиль':'Профиль исходного каталога'}</span><span>${escape(card.id)}</span><span>Цитата — сведения из профиля</span></div></article>`).join('');
  }
  const labels = {busy_date:'После проверки даты',budget:'После проверки бюджета',format:'После проверки формата',duration:'После проверки длительности',language:'После проверки языка'};
  const sourceLabels = {llm:'Акценты выбрала LLM',cache:'Проверенный выбор из кэша',facts:'Факты профиля без LLM',timeout:'Факты профиля: таймаут LLM',unavailable:'Факты профиля: API недоступен',invalid_response:'Факты профиля: ответ LLM отклонён',error:'Факты профиля: ошибка объяснения'};
  document.querySelector('#trace-content').innerHTML = `<div class="trace-row"><span>В городе и категории</span><strong>${data.pool_size}</strong></div>` + Object.entries(data.trace.stage_counts).map(([key,count]) => `<div class="trace-row"><span>${escape(labels[key] || key)}</span><strong>${count}</strong></div>`).join('') + data.cards.map(card => `<div class="trace-row"><span>${escape(card.name)} · балл ранжирования</span><strong>${card.score.toFixed(3)}</strong></div>`).join('');
  trace.hidden = false;
  for (const card of data.cards) {
    const row = document.createElement('div');
    row.className = 'trace-row';
    const name = document.createElement('span');
    name.textContent = card.name;
    const source = document.createElement('span');
    source.textContent = sourceLabels[data.trace.explanation_sources?.[card.id]] || 'Источник не указан';
    row.append(name,source);
    document.querySelector('#trace-content').append(row);
  }
}
async function search(event) {
  event?.preventDefault();
  if (!form.reportValidity()) return;
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  const started = performance.now();
  const deadline = setTimeout(() => controller.abort(), 15000);
  const values = Object.fromEntries(new FormData(form));
  const query = {...values, budget_kzt:Number(values.budget_kzt),duration_h:values.duration_h ? Number(values.duration_h) : null,language:values.language || null,brief:values.brief.trim() || null};
  submit.disabled = true;
  submit.textContent = 'Подбираем…';
  panel.setAttribute('aria-busy','true');
  statusBox.textContent = 'Проверяем дату и условия, сравниваем подходящие профили…';
  results.innerHTML = '<div class="loading">Готовим ваш короткий список</div>';
  trace.hidden = true;
  timing.textContent = '';
  try {
    const data = await getJSON('/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(query),signal:controller.signal});
    if (activeRequest !== controller) return;
    showResponse(data);
    timing.textContent = `${((performance.now()-started)/1000).toFixed(1)} с`;
  } catch(error) {
    if (activeRequest !== controller) return;
    statusBox.innerHTML = `<div class="notice error" role="alert">${escape(error.name === 'AbortError' ? 'Ответ занимает слишком много времени. Повторите запрос.' : error.message)}</div>`;
    results.textContent = '';
  } finally {
    clearTimeout(deadline);
    if (activeRequest === controller) {
      submit.disabled = false;
      submit.innerHTML = 'Подобрать подрядчиков <span aria-hidden="true">↗</span>';
      panel.setAttribute('aria-busy','false');
    }
  }
}
form.addEventListener('submit',search);
async function initialize() {
  try {
    const [catalog,scenarios] = await Promise.all([getJSON('/catalog'),getJSON('/demo-scenarios')]);
    for (const [name,key] of [['city','cities'],['category','categories'],['event_type','event_types'],['language','languages']]) {
      const select = form.elements.namedItem(name);
      if (name === 'language') select.add(new Option('Любой',''));
      catalog[key].forEach(value => select.add(new Option(value,value)));
    }
    form.elements.date.min = catalog.calendar_start;
    form.elements.date.max = catalog.calendar_end;
    document.querySelector('#calendar-hint').textContent = `Календарь: ${catalog.calendar_start} — ${catalog.calendar_end}`;
    document.querySelector('#catalog-info').textContent = `${catalog.profile_count} профилей в каталоге · ${catalog.categories.length} категорий · ${catalog.synthetic_count} синтетических профилей, отмеченных в выдаче`;
    if (scenarios.length) fill(scenarios[0].query);
    for (const scenario of scenarios) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = scenario.title;
      button.addEventListener('click',() => {fill(scenario.query);search();});
      document.querySelector('#scenarios').append(button);
    }
    submit.disabled = false;
  } catch(error) {
    statusBox.innerHTML = `<div class="notice error" role="alert">Не удалось загрузить каталог. Обновите страницу после запуска сервиса.</div>`;
    document.querySelector('#catalog-info').textContent = 'Каталог недоступен';
  }
}
initialize();
