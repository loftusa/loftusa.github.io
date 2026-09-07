import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./tests',testMatch:'*.spec.mjs',fullyParallel:false,
  use:{baseURL:'http://127.0.0.1:8086',headless:true,launchOptions:{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,args:['--use-angle=swiftshader','--enable-webgl']}},
  webServer:{command:'python3 -m http.server 8086 --bind 127.0.0.1 --directory ../../public/camp-sherman',url:'http://127.0.0.1:8086',reuseExistingServer:false},
});
