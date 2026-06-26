// middleware.ts
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(req: NextRequest) {
  if (req.nextUrl.pathname.startsWith("/api/admin/sync")) {
    const token = req.nextUrl.searchParams.get("secret");
    if (!token || token !== process.env.SYNC_TOKEN) {
      return new NextResponse("Unauthorized", { status: 401 });
    }
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/api/admin/sync"],
};

