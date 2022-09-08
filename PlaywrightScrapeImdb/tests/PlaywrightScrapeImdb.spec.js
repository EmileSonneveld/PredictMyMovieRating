// @ts-check
const { test, expect } = require('@playwright/test');
//npm i csv --save

const fs = require('fs')
let parse = require('csv-parse')

// npx playwright test

let imdb_to_metacritic = {}
const imdb_to_metacritic_path = "imdb_to_metacritic.json"
if (fs.existsSync(imdb_to_metacritic_path)) {
  imdb_to_metacritic = JSON.parse(fs.readFileSync(imdb_to_metacritic_path, {encoding:'utf8', flag:'r'}))
}
async function scrape(page, rows){
  for(let row of rows){
    console.log(row["URL"]);
    if(imdb_to_metacritic[row["Const"]] != null){
      continue;
    }
    await new Promise(r => setTimeout(r, Math.random()*1000));
    await page.goto(row["URL"]);

    const el = await page.$$("span.score > span");
    let score = ""
    if(el.length > 0){
      score = parseFloat(await page.locator('span.score > span').innerText());
    }
    console.log("score", score);
    imdb_to_metacritic[row["Const"]] = score
    const data = JSON.stringify(imdb_to_metacritic, null, 4);
    fs.writeFileSync(imdb_to_metacritic_path, data);
  }
}

test('scrape IMDB', async ({ page }) => {
  test.setTimeout(30 * 1000 * 99);
  await new Promise(resolve => {
    fs.readFile("../ratings.csv", function (err, fileData) {
      parse.parse(fileData, {columns: true, trim: true}, function(err, rows) {
        // Your CSV data is in an array of arrys passed to this callback as rows.
        scrape(page, rows).then(()=>resolve())
      })
    })
  })
})
