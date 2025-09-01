// app/api/admin/sync/route.ts
import { NextResponse } from "next/server";
import { syncSchwabHoldings } from "../../../../lib/schwabSync";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const secret = searchParams.get("secret");
  if (!secret || secret !== process.env.SYNC_TOKEN) {
    return new NextResponse("Unauthorized", { status: 401 });
  }

  await syncSchwabHoldings();
  return NextResponse.json({ ok: true });
}
