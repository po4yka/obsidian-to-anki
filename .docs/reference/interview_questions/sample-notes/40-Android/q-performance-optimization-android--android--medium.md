---
topic: android
tags:
  - android
  - performance
  - optimization
  - performance-rendering
  - performance-memory
  - checklist
difficulty: medium
status: draft
---

# App Performance Optimization Checklist for Android

**Сложность**: 🟡 Medium
**Источник**: Amit Shekhar Android Interview Questions

# Question (EN)
>

# Вопрос (RU)
>

---

## Answer (EN)
# Question (EN)
What is a comprehensive checklist for optimizing Android app performance? What are the key areas to focus on?

## Answer (EN)
Performance optimization requires a systematic approach across multiple areas. Here's a comprehensive checklist covering startup, runtime, memory, network, and rendering performance.

#### 1. **App Startup Optimization**

```kotlin
// - Application class optimization
class OptimizedApplication : Application() {
    override fun onCreate() {
        super.onCreate()

        // - DO: Critical initialization only
        initializeCrashReporting()

        // - DO: Defer non-critical work
        lifecycleScope.launch {
            initializeAnalytics()
            initializeAdSDK()
        }

        // - DON'T: Heavy work on main thread
        // loadConfiguration() // Blocking I/O
        // initializeAllLibraries() // Too much work
    }

    // - Use App Startup library
    // Automatically initializes dependencies in correct order
}

// - Lazy initialization
class LazyComponents {
    val database by lazy { createDatabase() }
    val imageLoader by lazy { ImageLoader.create() }
    val analytics by lazy { Analytics.initialize() }
}

// - Splash screen with windowBackground
// res/values/styles.xml
<style name="SplashTheme" parent="Theme.App">
    <item name="android:windowBackground">@drawable/splash</item>
</style>

// - Use Baseline Profiles
// Pre-compile critical startup code
// See q-baseline-profiles-android--android--medium.md
```

**Checklist:**
- [ ] Profile Application.onCreate() time
- [ ] Defer non-critical initialization
- [ ] Use App Startup library
- [ ] Implement baseline profiles
- [ ] Measure with `adb shell am start -W`
- [ ] Target: Cold start < 1000ms
- [ ] Use lazy initialization
- [ ] Splash screen via theme (not Activity)

#### 2. **UI Rendering Optimization**

```kotlin
// - Flatten view hierarchy
// - BAD: Deep nesting
<LinearLayout>
    <RelativeLayout>
        <FrameLayout>
            <LinearLayout>
                <TextView />
            </LinearLayout>
        </FrameLayout>
    </RelativeLayout>
</LinearLayout>

// - GOOD: Flat hierarchy with ConstraintLayout
<ConstraintLayout>
    <TextView
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent" />
</ConstraintLayout>

// - BEST: Jetpack Compose
@Composable
fun OptimizedScreen() {
    Column {
        Text("Hello")
        Button(onClick = {}) {
            Text("Click")
        }
    }
}

// - Avoid overdraw
class OverdrawOptimization {
    // - Multiple backgrounds drawing over each other
    // <LinearLayout android:background="@color/white">
    //     <View android:background="@color/gray" />
    // </LinearLayout>

    // - Remove unnecessary backgrounds
    // <LinearLayout> <!-- No background -->
    //     <View android:background="@color/gray" />
    // </LinearLayout>
}

// - ViewStub for rarely shown views
<ViewStub
    android:id="@+id/stub_rarely_used"
    android:layout="@layout/rarely_used"
    android:inflateId="@+id/rarely_used" />

// Inflate only when needed
binding.stubRarelyUsed.inflate()

// - Merge tag to eliminate redundant layouts
<merge xmlns:android="...">
    <TextView ... />
    <Button ... />
</merge>
```

**Checklist:**
- [ ] Enable "Profile GPU Rendering"
- [ ] Enable "Debug GPU Overdraw"
- [ ] Keep view hierarchy depth < 10
- [ ] Use ConstraintLayout or Compose
- [ ] Avoid overdraw (< 2x ideal)
- [ ] Target 60 FPS (16ms per frame)
- [ ] Use ViewStub for conditional views
- [ ] Merge layouts where possible

#### 3. **RecyclerView Optimization**

```kotlin
class OptimizedAdapter : ListAdapter<Item, ViewHolder>(DIFF_CALLBACK) {

    init {
        // - Enable stable IDs
        setHasStableIds(true)
    }

    override fun getItemId(position: Int): Long {
        return getItem(position).id
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        // - Use ViewBinding (faster than findViewById)
        val binding = ItemBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        // - Keep lightweight - no heavy operations
        holder.bind(getItem(position))
    }

    class ViewHolder(private val binding: ItemBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(item: Item) {
            binding.title.text = item.title

            // - Async image loading
            Glide.with(binding.root.context)
                .load(item.imageUrl)
                .placeholder(R.drawable.placeholder)
                .into(binding.image)
        }
    }

    companion object {
        private val DIFF_CALLBACK = object : DiffUtil.ItemCallback<Item>() {
            override fun areItemsTheSame(old: Item, new: Item) = old.id == new.id
            override fun areContentsTheSame(old: Item, new: Item) = old == new
        }
    }
}

// - Configure RecyclerView
recyclerView.apply {
    layoutManager = LinearLayoutManager(context).apply {
        isItemPrefetchEnabled = true
        initialPrefetchItemCount = 4
    }

    setHasFixedSize(true)
    setItemViewCacheSize(20)

    // - Shared RecycledViewPool for nested RecyclerViews
    setRecycledViewPool(sharedViewPool)
}

// - Use Paging 3 for large lists
val pager = Pager(
    config = PagingConfig(pageSize = 50, prefetchDistance = 10),
    pagingSourceFactory = { repository.getPagingSource() }
).flow
```

**Checklist:**
- [ ] Use ListAdapter with DiffUtil
- [ ] Enable stable IDs
- [ ] Use ViewBinding
- [ ] Keep onBindViewHolder lightweight
- [ ] Enable item prefetch
- [ ] Set appropriate cache size
- [ ] Use Paging 3 for large datasets
- [ ] Avoid nested RecyclerViews (use ConcatAdapter)

#### 4. **Memory Optimization**

```kotlin
// - Image optimization
class ImageOptimization {
    fun loadImage(imageView: ImageView, url: String) {
        Glide.with(imageView.context)
            .load(url)
            .override(imageView.width, imageView.height) // - Resize
            .diskCacheStrategy(DiskCacheStrategy.ALL)   // - Cache
            .placeholder(R.drawable.placeholder)
            .error(R.drawable.error)
            .into(imageView)
    }

    // - Bitmap sampling for large images
    suspend fun loadLargeBitmap(path: String, reqWidth: Int, reqHeight: Int): Bitmap {
        return withContext(Dispatchers.IO) {
            BitmapFactory.Options().run {
                inJustDecodeBounds = true
                BitmapFactory.decodeFile(path, this)

                inSampleSize = calculateInSampleSize(this, reqWidth, reqHeight)
                inJustDecodeBounds = false

                BitmapFactory.decodeFile(path, this)
            }
        }
    }
}

// - Leak prevention
class LeakPrevention : AppCompatActivity() {
    // - Use ViewModel for data retention
    private val viewModel: MyViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // - Lifecycle-aware observers
        viewModel.data.observe(this) { data ->
            updateUI(data)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // - Clean up resources
        cleanup()
    }
}

// - Use LeakCanary
// debugImplementation("com.squareup.leakcanary:leakcanary-android:2.12")

// - Memory-efficient collections
class CollectionOptimization {
    // - HashMap for primitive keys
    val map = HashMap<Int, String>()

    // - SparseArray for int keys
    val sparse = SparseIntArray()

    // - ArrayMap for small maps (<1000 items)
    val arrayMap = ArrayMap<String, String>()
}
```

**Checklist:**
- [ ] Use image loading libraries (Glide/Coil)
- [ ] Sample large bitmaps
- [ ] Fix memory leaks (LeakCanary)
- [ ] Use ViewModel for data retention
- [ ] Use SparseArray for primitive keys
- [ ] Implement proper cleanup
- [ ] Monitor memory usage
- [ ] Target: < 50MB RAM usage

#### 5. **Database Optimization**

```kotlin
@Dao
interface OptimizedDao {
    // - Use indices
    @Query("SELECT * FROM users WHERE email = :email")
    suspend fun getUserByEmail(email: String): User?

    // - Select only needed columns
    @Query("SELECT id, name FROM users")
    suspend fun getUserNames(): List<UserName>

    // - Use pagination
    @Query("SELECT * FROM users ORDER BY id")
    fun getUsersPaged(): PagingSource<Int, User>

    // - Batch operations
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(users: List<User>)

    // - Use transactions
    @Transaction
    suspend fun complexOperation() {
        // Multiple operations in single transaction
    }
}

@Entity(
    tableName = "users",
    indices = [
        Index(value = ["email"], unique = true),
        Index(value = ["last_name", "first_name"])
    ]
)
data class User(...)

// - Enable WAL mode (default in Room)
val db = Room.databaseBuilder(context, AppDatabase::class.java, "db")
    .setJournalMode(JournalMode.WRITE_AHEAD_LOGGING)
    .build()
```

**Checklist:**
- [ ] Add appropriate indices
- [ ] Use projections (select specific columns)
- [ ] Implement pagination
- [ ] Batch operations
- [ ] Use transactions
- [ ] Enable WAL mode
- [ ] Avoid main thread operations
- [ ] Profile queries (enable logging)

#### 6. **Network Optimization**

```kotlin
// - Efficient API calls
interface ApiService {
    // - Request only needed data
    @GET("users/{id}")
    suspend fun getUser(
        @Path("id") id: String,
        @Query("fields") fields: String = "id,name,email" // - Field filtering
    ): User

    // - Pagination
    @GET("articles")
    suspend fun getArticles(
        @Query("page") page: Int,
        @Query("pageSize") pageSize: Int = 50
    ): List<Article>

    // - Compression
    // OkHttp automatically handles gzip
}

// - Configure OkHttp
val client = OkHttpClient.Builder()
    .cache(Cache(context.cacheDir, 10L * 1024 * 1024)) // - 10MB cache
    .connectTimeout(30, TimeUnit.SECONDS)
    .readTimeout(30, TimeUnit.SECONDS)
    .addInterceptor(CachingInterceptor()) // - Cache responses
    .build()

// - Implement caching
class CachingInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val response = chain.proceed(request)

        return response.newBuilder()
            .header("Cache-Control", "public, max-age=300") // - 5 min cache
            .build()
    }
}

// - Offline-first architecture
class OfflineFirstRepository(
    private val apiService: ApiService,
    private val dao: ArticleDao
) {
    fun getArticles(): Flow<List<Article>> = flow {
        // - Emit cached data first
        emit(dao.getAllArticles())

        // - Fetch from network
        try {
            val articles = apiService.getArticles()
            dao.insertAll(articles)
            emit(dao.getAllArticles())
        } catch (e: Exception) {
            // - Continue with cached data
        }
    }
}
```

**Checklist:**
- [ ] Implement response caching
- [ ] Use pagination for lists
- [ ] Compress requests/responses
- [ ] Field filtering (request only needed data)
- [ ] Implement offline-first
- [ ] Batch network requests
- [ ] Use WorkManager for background sync
- [ ] Monitor network usage

#### 7. **Battery Optimization**

```kotlin
// - Use WorkManager for background tasks
class SyncWorker(context: Context, params: WorkerParameters) :
    CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        return try {
            syncData()
            Result.success()
        } catch (e: Exception) {
            Result.retry()
        }
    }
}

// - Schedule with constraints
val constraints = Constraints.Builder()
    .setRequiredNetworkType(NetworkType.CONNECTED)
    .setRequiresBatteryNotLow(true)
    .setRequiresCharging(false)
    .build()

val syncWork = PeriodicWorkRequestBuilder<SyncWorker>(15, TimeUnit.MINUTES)
    .setConstraints(constraints)
    .build()

WorkManager.getInstance(context).enqueue(syncWork)

// - Location updates optimization
fusedLocationClient.requestLocationUpdates(
    LocationRequest.create().apply {
        interval = 60_000 // - 1 minute (not every second)
        fastestInterval = 30_000
        priority = LocationRequest.PRIORITY_BALANCED_POWER_ACCURACY // - Not HIGH_ACCURACY
    },
    locationCallback,
    Looper.getMainLooper()
)

// - Use JobScheduler/WorkManager instead of AlarmManager
// - Batch network requests
// - Avoid wakelocks when possible
```

**Checklist:**
- [ ] Use WorkManager for background tasks
- [ ] Set appropriate constraints
- [ ] Batch network requests
- [ ] Optimize location updates
- [ ] Minimize wakelock usage
- [ ] Use Doze-aware APIs
- [ ] Profile with Battery Historian

#### 8. **Build Optimization**

```kotlin
// build.gradle.kts

android {
    // - Enable R8
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    // - Enable build cache
    buildCache {
        local {
            isEnabled = true
        }
    }

    // - Split APKs by ABI
    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
            isUniversalApk = false
        }
    }

    // - Use Android App Bundle
    // Automatically splits by density, language, ABI
}

// - Dependency optimization
dependencies {
    // - Avoid bloated libraries
    // implementation("com.google.guava:guava:31.1-android")

    // - Use lightweight alternatives
    implementation("androidx.collection:collection-ktx:1.3.0")
}
```

**Checklist:**
- [ ] Enable R8/ProGuard
- [ ] Shrink resources
- [ ] Use Android App Bundle
- [ ] Remove unused dependencies
- [ ] Split APKs by ABI
- [ ] Optimize images (WebP)
- [ ] Enable build cache
- [ ] Target: APK < 20MB

### Monitoring and Profiling

```kotlin
// - Performance monitoring
class PerformanceMonitor {
    fun trackPerformance() {
        // Firebase Performance
        val trace = Firebase.performance.newTrace("screen_load")
        trace.start()
        // ... operation
        trace.stop()

        // Custom metrics
        Firebase.analytics.logEvent("performance") {
            param("startup_time", startupTimeMs)
            param("memory_mb", memoryUsageMB)
        }
    }
}
```

**Tools:**
- [ ] Android Profiler (CPU, Memory, Network, Energy)
- [ ] Layout Inspector
- [ ] LeakCanary
- [ ] StrictMode (debug builds)
- [ ] Firebase Performance Monitoring
- [ ] Perfetto/Systrace
- [ ] Baseline Profiles

### Performance Targets

| Metric | Target | Good | Acceptable |
|--------|--------|------|------------|
| Cold Start | < 500ms | < 1000ms | < 1500ms |
| Frame Rate | 60 FPS | 50-60 FPS | > 45 FPS |
| Memory | < 50MB | < 100MB | < 150MB |
| APK Size | < 10MB | < 20MB | < 50MB |
| Battery/Hour | < 3% | < 5% | < 10% |

---



## Ответ (RU)
# Вопрос (RU)
Каков комплексный чек-лист оптимизации производительности Android-приложения? На какие ключевые области сосредоточиться?

## Ответ (RU)
Оптимизация производительности требует системного подхода.

#### Основные области:

**1. Запуск приложения:**
- Отложенная инициализация
- App Startup library
- Baseline Profiles
- Цель: < 1000ms

**2. UI рендеринг:**
- Плоская иерархия view
- ConstraintLayout/Compose
- Избегать overdraw
- Цель: 60 FPS

**3. RecyclerView:**
- ListAdapter + DiffUtil
- ViewBinding
- Item prefetch
- Paging 3

**4. Память:**
- Image libraries (Glide/Coil)
- LeakCanary
- ViewModel
- Цель: < 50MB RAM

**5. База данных:**
- Индексы
- Пагинация
- Batch операции
- WAL mode

**6. Сеть:**
- Кэширование
- Offline-first
- Пагинация
- Сжатие

**7. Батарея:**
- WorkManager
- Оптимизация location
- Batch requests

**8. Сборка:**
- R8/ProGuard
- App Bundle
- Resource shrinking
- Цель: < 20MB APK

#### Инструменты:
- Android Profiler
- LeakCanary
- Firebase Performance
- Perfetto

Системная оптимизация всех областей даёт наилучший результат.
