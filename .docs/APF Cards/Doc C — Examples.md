**Simple**

```html
<!-- Card 1 | slug: kotlin-context-merge | CardType: Simple | Tags: kotlin coroutines flow -->

<!-- Title -->
Which Kotlin operator merges two CoroutineContexts and which side wins conflicts?

<!-- Sample (code block) -->
<pre><code class="language-kotlin">val ctx = Dispatchers.Main + Dispatchers.IO</code></pre>

<!-- Key point (code block) -->
<pre><code class="language-kotlin">// Right-biased: the rightmost element with the same Key wins.
Dispatchers.IO</code></pre>

<!-- Key point notes -->
<ul>
  <li>Plus folds elements; RHS with same Key replaces LHS.</li>
  <li>Both dispatchers share Key ContinuationInterceptor.</li>
  <li>Result launches on IO.</li>
</ul>

<!-- manifest: {"slug":"kotlin-context-merge","lang":"kotlin","type":"Simple","tags":["kotlin","coroutines","flow"]} -->
```

**Missing**

```html
<!-- Card 2 | slug: gha-gradle-cache-basics | CardType: Missing | Tags: ci_cd github_actions gradle_cache -->

<!-- Title -->
Complete the cache setup for Gradle in GitHub Actions.

<!-- Key point (code block with cloze) -->
<pre><code class="language-yaml">- name: Cache Gradle
  uses: actions/cache@v{{c1::3}}
  with:
    path: ~/.gradle/caches
    key: gradle-{{c2::${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }} }}</code></pre>

<!-- Key point notes -->
<ul>
  <li>v3 supports larger caches and cleanup.</li>
  <li>Key hashes Gradle files to avoid stale deps.</li>
</ul>

<!-- manifest: {"slug":"gha-gradle-cache-basics","lang":"yaml","type":"Missing","tags":["ci_cd","github_actions","gradle_cache"]} -->
```

**Draw**

```html
<!-- Card 3 | slug: retrofit-call-flow | CardType: Draw | Tags: android architecture retrofit -->

<!-- Title -->
Sketch the call flow from ViewModel.getUser() to the HTTP socket.

<!-- Key point (image) -->
<img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='320'><rect width='100%' height='100%' fill='white'/><g font-family='monospace' font-size='14'><text x='20' y='30'>UI/ViewModel  UseCase  Repository  RemoteDataSource</text><text x='20' y='60'> Retrofit proxy  OkHttp interceptors  Transport  Server</text></g></svg>" alt="sequence diagram"/>

<!-- Key point notes -->
<ul>
  <li>Suspension before I/O; resumes on response.</li>
  <li>Interceptors: logging, auth, caching.</li>
</ul>

<!-- manifest: {"slug":"retrofit-call-flow","lang":"svg","type":"Draw","tags":["android","architecture","retrofit"]} -->
```
